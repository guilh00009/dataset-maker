//! A mock OpenAI-compatible LLM server for developing and testing the translation pipeline.
//!
//! It speaks the same HTTP API as llama.cpp's `llama-server` (`GET /v1/models`,
//! `POST /v1/chat/completions`, `GET /props`), so the app talks to it exactly as it would to
//! a real server: there is no mock-specific code path in the application. It "translates"
//! English into Portuguese with a small built-in dictionary and can simulate slots, latency,
//! rate limits (429 + Retry-After), server errors, timeouts, malformed JSON and missing items.

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;
use tokio::sync::Semaphore;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MockConfig {
    pub model: String,
    /// When set, requests must carry `Authorization: Bearer <api_key>`.
    pub api_key: Option<String>,
    /// Parallel slots (like `llama-server --parallel`).
    pub slots: usize,
    pub base_latency_ms: u64,
    /// Simulated generation speed: delay per output token.
    pub ms_per_token: u64,
    pub rate_429: f64,
    pub retry_after_secs: f64,
    pub rate_500: f64,
    /// Probability that a request hangs for an hour (client timeout test).
    pub rate_timeout: f64,
    /// Probability of answering with non-JSON garbage in JSON mode.
    pub rate_malformed: f64,
    /// Probability of dropping one item from a batched answer.
    pub rate_drop_item: f64,
    /// Probability of returning an English (untranslated) answer.
    pub rate_untranslated: f64,
    /// Behave like an older server that rejects `response_format`.
    pub reject_response_format: bool,
    pub seed: u64,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            model: "mock-translator-7b".into(),
            api_key: None,
            slots: 4,
            base_latency_ms: 20,
            ms_per_token: 1,
            rate_429: 0.0,
            retry_after_secs: 0.2,
            rate_500: 0.0,
            rate_timeout: 0.0,
            rate_malformed: 0.0,
            rate_drop_item: 0.0,
            rate_untranslated: 0.0,
            reject_response_format: false,
            seed: 1,
        }
    }
}

#[derive(Debug, Default, Serialize)]
pub struct MockStats {
    pub requests: u64,
    pub ok: u64,
    pub sent_429: u64,
    pub sent_500: u64,
    pub unauthorized: u64,
    pub rejected_response_format: u64,
    pub max_in_flight: u64,
    pub items_translated: u64,
}

pub struct MockState {
    cfg: RwLock<MockConfig>,
    slots: Semaphore,
    counter: AtomicU64,
    in_flight: AtomicU64,
    requests: AtomicU64,
    ok: AtomicU64,
    sent_429: AtomicU64,
    sent_500: AtomicU64,
    unauthorized: AtomicU64,
    rejected_rf: AtomicU64,
    max_in_flight: AtomicU64,
    items: AtomicU64,
    log: RwLock<Vec<String>>,
}

impl MockState {
    pub fn new(cfg: MockConfig) -> Arc<Self> {
        Arc::new(Self {
            slots: Semaphore::new(cfg.slots.max(1)),
            cfg: RwLock::new(cfg),
            counter: AtomicU64::new(0),
            in_flight: AtomicU64::new(0),
            requests: AtomicU64::new(0),
            ok: AtomicU64::new(0),
            sent_429: AtomicU64::new(0),
            sent_500: AtomicU64::new(0),
            unauthorized: AtomicU64::new(0),
            rejected_rf: AtomicU64::new(0),
            max_in_flight: AtomicU64::new(0),
            items: AtomicU64::new(0),
            log: RwLock::new(Vec::new()),
        })
    }

    pub fn stats(&self) -> MockStats {
        MockStats {
            requests: self.requests.load(Ordering::SeqCst),
            ok: self.ok.load(Ordering::SeqCst),
            sent_429: self.sent_429.load(Ordering::SeqCst),
            sent_500: self.sent_500.load(Ordering::SeqCst),
            unauthorized: self.unauthorized.load(Ordering::SeqCst),
            rejected_response_format: self.rejected_rf.load(Ordering::SeqCst),
            max_in_flight: self.max_in_flight.load(Ordering::SeqCst),
            items_translated: self.items.load(Ordering::SeqCst),
        }
    }

    pub fn set_config(&self, cfg: MockConfig) {
        *self.cfg.write().unwrap() = cfg;
    }

    pub fn config(&self) -> MockConfig {
        self.cfg.read().unwrap().clone()
    }

    /// Deterministic pseudo-random event with probability `p`.
    fn chance(&self, p: f64) -> bool {
        if p <= 0.0 {
            return false;
        }
        if p >= 1.0 {
            return true;
        }
        let n = self.counter.fetch_add(1, Ordering::SeqCst);
        let mut z = n.wrapping_add(self.cfg.read().unwrap().seed).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 / (1u64 << 53) as f64) < p
    }

    /// Texts of the requests received so far (for assertions in tests).
    pub fn request_log(&self) -> Vec<String> {
        self.log.read().unwrap().clone()
    }
}

// ───────────────────────────── "translation" ─────────────────────────────

const DICT: &str = "the=o|a=um|an=um|and=e|or=ou|but=mas|of=de|to=para|in=em|on=em|at=em|with=com|for=para|from=de|by=por|is=é|are=está|was=era|were=eram|be=ser|been=sido|am=sou|i=eu|you=você|he=ele|she=ela|it=isso|we=nós|they=eles|me=me|him=ele|her=dela|us=nós|them=eles|my=meu|your=seu|his=dele|its=seu|our=nosso|their=deles|this=este|that=aquele|these=estes|those=aqueles|what=o quê|who=quem|where=onde|when=quando|why=por quê|how=como|which=qual|not=não|no=não|yes=sim|hello=olá|hi=oi|hey=ei|goodbye=adeus|bye=tchau|please=por favor|thanks=obrigado|thank=agradeço|sorry=desculpe|good=bom|bad=ruim|great=ótimo|nice=legal|very=muito|much=muito|many=muitos|more=mais|most=mais|some=algum|any=qualquer|all=todos|every=todo|each=cada|other=outro|new=novo|old=velho|big=grande|small=pequeno|little=pequeno|long=longo|short=curto|high=alto|low=baixo|first=primeiro|last=último|next=próximo|day=dia|night=noite|morning=manhã|evening=tarde|today=hoje|tomorrow=amanhã|yesterday=ontem|time=tempo|year=ano|week=semana|man=homem|woman=mulher|boy=menino|girl=menina|friend=amigo|people=pessoas|person=pessoa|child=criança|world=mundo|life=vida|love=amor|home=casa|house=casa|room=quarto|door=porta|window=janela|water=água|food=comida|tea=chá|coffee=café|flowers=flores|flower=flor|garden=jardim|shop=loja|store=loja|market=mercado|city=cidade|town=cidade|forest=floresta|river=rio|mountain=montanha|sun=sol|moon=lua|star=estrela|sky=céu|rain=chuva|wind=vento|fire=fogo|tree=árvore|book=livro|story=história|word=palavra|name=nome|question=pergunta|answer=resposta|help=ajuda|work=trabalho|play=jogar|game=jogo|music=música|song=canção|dance=dançar|smile=sorrir|smiles=sorri|laughs=ri|laugh=rir|looks=olha|look=olhar|see=ver|saw=viu|watch=observar|hear=ouvir|listen=escutar|say=dizer|says=diz|said=disse|tell=contar|told=contou|ask=perguntar|asks=pergunta|asked=perguntou|speak=falar|talk=conversar|go=ir|goes=vai|went=foi|come=vir|comes=vem|came=veio|take=pegar|took=pegou|give=dar|gave=deu|make=fazer|made=fez|do=fazer|does=faz|did=fez|have=ter|has=tem|had=teve|get=conseguir|got=conseguiu|know=saber|knew=sabia|think=pensar|thought=pensou|feel=sentir|felt=sentiu|want=querer|wants=quer|wanted=queria|need=precisar|needs=precisa|like=gostar|likes=gosta|loves=ama|try=tentar|tried=tentou|use=usar|find=encontrar|found=encontrou|keep=manter|let=deixar|begin=começar|start=começar|started=começou|stop=parar|wait=esperar|walk=caminhar|walks=caminha|run=correr|sit=sentar|open=abrir|opens=abre|close=fechar|read=ler|write=escrever|learn=aprender|teach=ensinar|buy=comprar|sell=vender|pay=pagar|eat=comer|drink=beber|sleep=dormir|live=viver|lives=vive|die=morrer|can=posso|could=poderia|will=vai|would=gostaria|should=deveria|must=deve|may=pode|might=poderia|just=apenas|only=somente|also=também|too=também|still=ainda|already=já|again=novamente|always=sempre|never=nunca|sometimes=às vezes|often=frequentemente|maybe=talvez|perhaps=talvez|here=aqui|there=lá|now=agora|then=então|soon=logo|later=depois|before=antes|after=depois|about=sobre|around=por aí|through=através|over=sobre|under=sob|between=entre|into=em|out=fora|up=para cima|down=para baixo|off=fora|if=se|because=porque|so=então|as=como|than=do que|while=enquanto|until=até|welcome=bem-vindo|beautiful=lindo|pretty=bonito|happy=feliz|sad=triste|angry=bravo|afraid=com medo|tired=cansado|hungry=com fome|busy=ocupado|free=livre|ready=pronto|sure=certo|right=certo|wrong=errado|true=verdade|real=real|important=importante|special=especial|different=diferente|same=mesmo|own=próprio|well=bem|better=melhor|best=melhor|really=realmente|quite=bastante|almost=quase|enough=suficiente|together=juntos|alone=sozinho|without=sem|something=algo|nothing=nada|everything=tudo|everyone=todos|anyone=alguém|someone=alguém|way=caminho|place=lugar|thing=coisa|things=coisas|lot=muito|bit=pouco|kind=tipo|hand=mão|hands=mãos|eyes=olhos|eye=olho|face=rosto|head=cabeça|heart=coração|voice=voz|tonight=esta noite|customer=cliente|florist=florista|bouquet=buquê|roses=rosas|red=vermelho|blue=azul|green=verde|white=branco|black=preto|yellow=amarelo|i'm=eu estou|it's=é|that's=isso é|don't=não|can't=não posso|won't=não vou|i'll=eu vou|you're=você é|let's=vamos|there's=há|what's=o que é|today's=de hoje|doing=fazendo|going=indo|coming=vindo|being=sendo|having=tendo|looking=olhando|thinking=pensando|talking=conversando|working=trabalhando|living=vivendo|waiting=esperando|walking=caminhando|selling=vendendo|buying=comprando|helping=ajudando|trying=tentando|making=fazendo|taking=pegando|giving=dando|finding=encontrando|asking=perguntando|telling=contando|knowing=sabendo|feeling=sentindo|wanting=querendo|needing=precisando|loving=amando|liking=gostando|running=correndo|sitting=sentando|standing=de pé|opening=abrindo|closing=fechando|reading=lendo|writing=escrevendo|learning=aprendendo|teaching=ensinando|paying=pagando|eating=comendo|drinking=bebendo|sleeping=dormindo|dying=morrendo|started=começou|how's=como está|where's=onde está|who's=quem é|everything's=tudo está|nobody=ninguém|anything=qualquer coisa|else=mais|ever=já|even=até|though=embora|yet=ainda|both=ambos|either=qualquer|neither=nenhum|another=outro|such=tal|own=próprio|few=poucos|less=menos|least=menos|since=desde|during=durante|against=contra|among=entre|across=através|behind=atrás|beside=ao lado|beyond=além|inside=dentro|outside=fora|toward=em direção|upon=sobre|within=dentro|along=ao longo|past=passado|near=perto|far=longe|away=embora|back=de volta|ahead=à frente|forward=adiante|upstairs=lá em cima|downstairs=lá embaixo|really=realmente|quickly=rapidamente|slowly=devagar|softly=suavemente|gently=gentilmente|carefully=cuidadosamente|suddenly=de repente|finally=finalmente|actually=na verdade|honestly=honestamente|certainly=certamente|probably=provavelmente|definitely=definitivamente|exactly=exatamente|especially=especialmente|usually=geralmente|simply=simplesmente|clearly=claramente|slightly=levemente|deeply=profundamente|warmly=calorosamente|nervously=nervosamente|curiously=curiosamente|shyly=timidamente|brightly=brilhantemente|quietly=silenciosamente|loudly=alto|cheerfully=alegremente";

fn dict() -> &'static std::collections::HashMap<&'static str, &'static str> {
    static D: OnceLock<std::collections::HashMap<&'static str, &'static str>> = OnceLock::new();
    D.get_or_init(|| DICT.split('|').filter_map(|p| p.split_once('=')).collect())
}

/// Word-by-word English → Portuguese. Keeps punctuation, markdown and `⟦n⟧` tokens.
pub fn fake_translate(text: &str) -> String {
    static WORD: OnceLock<Regex> = OnceLock::new();
    let rx = WORD.get_or_init(|| Regex::new(r"[A-Za-z]+(?:'[A-Za-z]+)?").unwrap());
    rx.replace_all(text, |c: &regex::Captures| {
        let w = &c[0];
        match dict().get(w.to_lowercase().as_str()) {
            Some(t) => {
                if w.chars().next().is_some_and(|c| c.is_uppercase()) {
                    let mut cs = t.chars();
                    cs.next().map(|f| f.to_uppercase().collect::<String>() + cs.as_str()).unwrap_or_default()
                } else {
                    t.to_string()
                }
            }
            None => w.to_string(),
        }
    })
    .into_owned()
}

fn approx_tokens(s: &str) -> u64 {
    (s.chars().count() as f64 / 3.6).ceil() as u64
}

// ───────────────────────────── handlers ─────────────────────────────

fn authorized(state: &MockState, headers: &HeaderMap) -> bool {
    match &state.cfg.read().unwrap().api_key {
        None => true,
        Some(k) => headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.strip_prefix("Bearer ").is_some_and(|t| t == k)),
    }
}

fn err(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({"error": {"message": msg, "type": "mock_error", "code": status.as_u16()}}))).into_response()
}

async fn models(State(s): State<Arc<MockState>>, headers: HeaderMap) -> Response {
    if !authorized(&s, &headers) {
        s.unauthorized.fetch_add(1, Ordering::SeqCst);
        return err(StatusCode::UNAUTHORIZED, "Invalid API Key");
    }
    let cfg = s.config();
    Json(json!({"object": "list", "data": [{"id": cfg.model, "object": "model", "owned_by": "mock"}]})).into_response()
}

async fn props(State(s): State<Arc<MockState>>) -> Response {
    let cfg = s.config();
    Json(json!({"total_slots": cfg.slots, "model_path": format!("/models/{}.gguf", cfg.model), "default_generation_settings": {"n_ctx": 16384}}))
        .into_response()
}

async fn chat(State(s): State<Arc<MockState>>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    s.requests.fetch_add(1, Ordering::SeqCst);
    if !authorized(&s, &headers) {
        s.unauthorized.fetch_add(1, Ordering::SeqCst);
        return err(StatusCode::UNAUTHORIZED, "Invalid API Key");
    }
    let cfg = s.config();
    if cfg.reject_response_format && body.get("response_format").is_some() {
        s.rejected_rf.fetch_add(1, Ordering::SeqCst);
        return err(StatusCode::BAD_REQUEST, "unsupported parameter: response_format / json_schema");
    }
    if body.get("model").and_then(|m| m.as_str()).is_some_and(|m| !m.is_empty() && m != cfg.model) {
        return err(StatusCode::NOT_FOUND, &format!("model '{}' not found", body["model"].as_str().unwrap_or("")));
    }
    if s.chance(cfg.rate_429) {
        s.sent_429.fetch_add(1, Ordering::SeqCst);
        let mut r = err(StatusCode::TOO_MANY_REQUESTS, "rate limit exceeded");
        r.headers_mut().insert(header::RETRY_AFTER, format!("{}", cfg.retry_after_secs).parse().unwrap());
        return r;
    }
    if s.chance(cfg.rate_500) {
        s.sent_500.fetch_add(1, Ordering::SeqCst);
        return err(StatusCode::INTERNAL_SERVER_ERROR, "simulated server error");
    }
    if s.chance(cfg.rate_timeout) {
        tokio::time::sleep(Duration::from_secs(3600)).await;
    }

    // wait for a free slot, like llama-server does
    let _permit = s.slots.acquire().await.unwrap();
    let now = s.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
    s.max_in_flight.fetch_max(now, Ordering::SeqCst);
    struct Guard<'a>(&'a AtomicU64);
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let _g = Guard(&s.in_flight);

    let messages = body.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();
    let system = messages.iter().find(|m| m["role"] == "system").and_then(|m| m["content"].as_str()).unwrap_or("").to_string();
    let user = messages.iter().rev().find(|m| m["role"] == "user").and_then(|m| m["content"].as_str()).unwrap_or("").to_string();
    s.log.write().unwrap().push(user.clone());
    let prompt_tokens = approx_tokens(&system) + approx_tokens(&user);
    let max_tokens = body.get("max_tokens").and_then(|m| m.as_u64()).unwrap_or(1024);

    let untranslated = s.chance(cfg.rate_untranslated);
    let tr = |t: &str| if untranslated { t.to_string() } else { fake_translate(t) };

    let json_mode = system.contains("\"translations\"");
    let mut content: String;
    let mut items = 1u64;
    if json_mode {
        match serde_json::from_str::<Value>(&user) {
            Ok(v) => {
                let arr = v["translations"].as_array().cloned().unwrap_or_default();
                let mut out: Vec<Value> = arr.iter().map(|it| json!({"i": it["i"], "text": tr(it["text"].as_str().unwrap_or(""))})).collect();
                items = out.len() as u64;
                if out.len() > 1 && s.chance(cfg.rate_drop_item) {
                    out.remove(0);
                }
                content = json!({"translations": out}).to_string();
            }
            Err(_) => content = "I could not parse the request.".into(),
        }
        if s.chance(cfg.rate_malformed) {
            content = "Sure! Here are the translations: {{ not json".into();
        }
    } else {
        content = tr(&user);
    }
    s.items.fetch_add(items, Ordering::SeqCst);

    let mut completion_tokens = approx_tokens(&content);
    let mut finish = "stop";
    if completion_tokens > max_tokens {
        let keep = (max_tokens as f64 * 3.6) as usize;
        content = content.chars().take(keep).collect();
        completion_tokens = max_tokens;
        finish = "length";
    }
    let delay = cfg.base_latency_ms + cfg.ms_per_token * completion_tokens;
    tokio::time::sleep(Duration::from_millis(delay)).await;
    s.ok.fetch_add(1, Ordering::SeqCst);
    Json(json!({
        "id": format!("chatcmpl-mock-{}", s.requests.load(Ordering::SeqCst)),
        "object": "chat.completion",
        "model": cfg.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }))
    .into_response()
}

async fn mock_stats(State(s): State<Arc<MockState>>) -> Json<Value> {
    Json(serde_json::to_value(s.stats()).unwrap())
}

async fn mock_config(State(s): State<Arc<MockState>>, Json(v): Json<Value>) -> Response {
    match serde_json::from_value::<MockConfig>(v) {
        Ok(c) => {
            s.set_config(c);
            Json(json!({"ok": true})).into_response()
        }
        Err(e) => err(StatusCode::BAD_REQUEST, &e.to_string()),
    }
}

pub fn router(state: Arc<MockState>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/props", get(props))
        .route("/__mock/stats", get(mock_stats))
        .route("/__mock/config", post(mock_config))
        .with_state(state)
}

/// Start the mock on `addr` (use port 0 for a random free port). Returns the bound address.
pub async fn serve(cfg: MockConfig, addr: SocketAddr) -> anyhow::Result<(SocketAddr, Arc<MockState>, tokio::task::JoinHandle<()>)> {
    let state = MockState::new(cfg);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    let app = router(state.clone());
    let h = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok((bound, state, h))
}


// ───────────────────────────── mock Hugging Face Hub ─────────────────────────────

fn hf_root(dir: &std::path::Path, owner: &str, name: &str) -> PathBuf {
    dir.join(format!("{owner}__{name}"))
}

fn walk_files(root: &std::path::Path, dir: &std::path::Path, out: &mut Vec<Value>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk_files(root, &p, out);
        } else if let (Ok(rel), Ok(m)) = (p.strip_prefix(root), e.metadata()) {
            out.push(json!({"type": "file", "path": rel.to_string_lossy().replace('\\', "/"), "size": m.len()}));
        }
    }
}

async fn hf_tree(State(dir): State<PathBuf>, axum::extract::Path((owner, name, _rev)): axum::extract::Path<(String, String, String)>) -> Response {
    let root = hf_root(&dir, &owner, &name);
    if !root.exists() {
        return err(StatusCode::NOT_FOUND, "Repository not found");
    }
    let mut out = Vec::new();
    walk_files(&root, &root, &mut out);
    Json(out).into_response()
}

async fn hf_resolve(
    State(dir): State<PathBuf>,
    axum::extract::Path((owner, name, _rev, path)): axum::extract::Path<(String, String, String, String)>,
    headers: HeaderMap,
) -> Response {
    let file = hf_root(&dir, &owner, &name).join(&path);
    if path.contains("..") || !file.is_file() {
        return err(StatusCode::NOT_FOUND, "Entry not found");
    }
    let Ok(bytes) = std::fs::read(&file) else { return err(StatusCode::INTERNAL_SERVER_ERROR, "read error") };
    let total = bytes.len();
    let range = headers.get(header::RANGE).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("bytes=")).and_then(|v| v.split('-').next()).and_then(|v| v.parse::<usize>().ok());
    match range {
        Some(start) if start >= total => (StatusCode::RANGE_NOT_SATISFIABLE, [(header::CONTENT_RANGE, format!("bytes */{total}"))]).into_response(),
        Some(start) => (
            StatusCode::PARTIAL_CONTENT,
            [(header::CONTENT_RANGE, format!("bytes {start}-{}/{total}", total - 1)), (header::CONTENT_LENGTH, (total - start).to_string())],
            bytes[start..].to_vec(),
        )
            .into_response(),
        None => ([(header::CONTENT_LENGTH, total.to_string())], bytes).into_response(),
    }
}

pub fn hf_router(dir: PathBuf) -> Router {
    Router::new()
        .route("/api/datasets/{owner}/{name}/tree/{rev}", get(hf_tree))
        .route("/datasets/{owner}/{name}/resolve/{rev}/{*path}", get(hf_resolve))
        .with_state(dir)
}

/// Like [`serve`], and also serves a fake Hugging Face Hub from `hf_dir`
/// (files live in `<hf_dir>/<owner>__<name>/…`).
pub async fn serve_full(cfg: MockConfig, addr: SocketAddr, hf_dir: Option<PathBuf>) -> anyhow::Result<(SocketAddr, Arc<MockState>, tokio::task::JoinHandle<()>)> {
    let state = MockState::new(cfg);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    let mut app = router(state.clone());
    if let Some(d) = hf_dir {
        app = app.merge(hf_router(d));
    }
    let h = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok((bound, state, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_translation_keeps_structure() {
        assert_eq!(fake_translate("Hello, how are you?"), "Olá, como está você?");
        assert_eq!(fake_translate("*smiles* Welcome to the shop \u{27e6}0\u{27e7}!"), "*sorri* Bem-vindo para o loja \u{27e6}0\u{27e7}!");
    }
}
