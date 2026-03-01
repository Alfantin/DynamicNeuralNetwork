import { MODEL_OPTIONS, SELECTORS } from './config.js';
import { AppController } from './ui/AppController.js';

function getElements() {
  return Object.fromEntries(
    Object.entries(SELECTORS).map(([key, id]) => [key, document.getElementById(id)])
  );
}

const app = new AppController(getElements(), MODEL_OPTIONS);
app.init();
