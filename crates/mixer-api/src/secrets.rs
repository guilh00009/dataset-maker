//! API tokens live in the OS keychain when one is available, otherwise in a `0600` file in the
//! config directory. They are never written to recipes, logs or exports.

use std::path::PathBuf;

const SERVICE: &str = "dataset-mixer";

pub struct Secrets {
    file: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Store {
    Keychain,
    File,
}

impl Secrets {
    pub fn new(config_dir: &std::path::Path) -> Self {
        Self { file: config_dir.join("secrets.json") }
    }

    fn read_file(&self) -> serde_json::Map<String, serde_json::Value> {
        std::fs::read(&self.file).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    fn write_file(&self, m: &serde_json::Map<String, serde_json::Value>) -> std::io::Result<()> {
        if let Some(p) = self.file.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(&self.file, serde_json::to_vec_pretty(m).unwrap_or_default())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&self.file, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }

    /// Environment variables win (handy for scripts): `MIXER_LLM_TOKEN`, `HF_TOKEN`.
    fn env_name(name: &str) -> Option<&'static str> {
        match name {
            "llm_token" => Some("MIXER_LLM_TOKEN"),
            "hf_token" => Some("HF_TOKEN"),
            _ => None,
        }
    }

    pub fn get(&self, name: &str) -> Option<String> {
        if let Some(v) = Self::env_name(name).and_then(|e| std::env::var(e).ok()).filter(|v| !v.trim().is_empty()) {
            return Some(v);
        }
        if let Ok(e) = keyring::Entry::new(SERVICE, name) {
            if let Ok(v) = e.get_password() {
                if !v.is_empty() {
                    return Some(v);
                }
            }
        }
        self.read_file().get(name).and_then(|v| v.as_str()).map(|s| s.to_string()).filter(|s| !s.is_empty())
    }

    /// Store (or clear, with `None`). Returns where the value ended up.
    pub fn set(&self, name: &str, value: Option<&str>) -> Store {
        let value = value.map(str::trim).filter(|v| !v.is_empty());
        let mut file = self.read_file();
        match value {
            Some(v) => {
                if let Ok(e) = keyring::Entry::new(SERVICE, name) {
                    if e.set_password(v).is_ok() && e.get_password().map(|g| g == v).unwrap_or(false) {
                        file.remove(name);
                        let _ = self.write_file(&file);
                        return Store::Keychain;
                    }
                }
                file.insert(name.into(), v.into());
                let _ = self.write_file(&file);
                Store::File
            }
            None => {
                if let Ok(e) = keyring::Entry::new(SERVICE, name) {
                    let _ = e.delete_credential();
                }
                file.remove(name);
                let _ = self.write_file(&file);
                Store::File
            }
        }
    }

    pub fn has(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Where a stored secret currently lives, for display.
    pub fn location(&self, name: &str) -> Option<Store> {
        if keyring::Entry::new(SERVICE, name).ok().and_then(|e| e.get_password().ok()).is_some_and(|v| !v.is_empty()) {
            return Some(Store::Keychain);
        }
        if self.read_file().contains_key(name) {
            return Some(Store::File);
        }
        None
    }
}
