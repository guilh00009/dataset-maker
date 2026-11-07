// Import order required by ViraUI: fonts -> preflight -> theme.
import '@fontsource-variable/geist/wght.css';
import '@fontsource-variable/geist-mono/wght.css';
import '@viraui/react/preflight.css';
import '@viraui/foundation/vira.css';
import './app.css';

import { createRoot } from 'react-dom/client';
import { Dialog, Toast, Tooltip } from '@viraui/react';
import { App } from './App';
import { StoreProvider, toastManager } from './store';

function Toasts() {
  const { toasts } = Toast.useToastManager();
  return (
    <Toast.Portal>
      <Toast.Viewport position="bottom-right">
        {toasts.map((t) => (
          <Toast.Banner key={t.id} toast={t} description={t.description} />
        ))}
      </Toast.Viewport>
    </Toast.Portal>
  );
}

createRoot(document.getElementById('root')!).render(
  <Dialog.Provider>
    <Dialog.IndentBackground>
      <Dialog.Indent>
        <Tooltip.Provider>
          <Toast.Provider toastManager={toastManager}>
            <StoreProvider>
              <App />
            </StoreProvider>
            <Toasts />
          </Toast.Provider>
        </Tooltip.Provider>
      </Dialog.Indent>
    </Dialog.IndentBackground>
  </Dialog.Provider>,
);
