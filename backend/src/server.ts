import express, { Request, Response } from 'express';
import cors from 'cors';
import Anthropic from '@anthropic-ai/sdk';

const app = express();
const client = new Anthropic();

if (!process.env.ANTHROPIC_API_KEY) {
  console.warn('⚠️  ANTHROPIC_API_KEY is not set. Set it before starting the server.');
}

app.use(cors({ origin: ['http://localhost:5173', 'http://localhost:4173'] }));
app.use(express.json());

const MODELS = [
  { id: 'claude-opus-4-6', name: 'Claude Opus 4.6' },
  { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' },
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5' },
];

const VALID_MODEL_IDS = new Set(MODELS.map((m) => m.id));

app.get('/api/models', (_req: Request, res: Response) => {
  res.json(MODELS);
});

interface ChatBody {
  messages: Array<{ role: 'user' | 'assistant'; content: string }>;
  model?: string;
  system?: string;
}

app.post('/api/chat', async (req: Request, res: Response) => {
  const { messages, model = 'claude-opus-4-6', system } = req.body as ChatBody;

  if (!Array.isArray(messages) || messages.length === 0) {
    res.status(400).json({ error: 'messages array is required' });
    return;
  }

  if (!VALID_MODEL_IDS.has(model)) {
    res.status(400).json({ error: `Invalid model: ${model}` });
    return;
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (payload: object) => {
    res.write(`data: ${JSON.stringify(payload)}\n\n`);
  };

  try {
    const stream = client.messages.stream({
      model,
      max_tokens: 4096,
      ...(system?.trim() ? { system: system.trim() } : {}),
      messages,
    });

    for await (const event of stream) {
      if (
        event.type === 'content_block_delta' &&
        event.delta.type === 'text_delta'
      ) {
        send({ type: 'text', text: event.delta.text });
      }
    }

    send({ type: 'done' });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unexpected server error';
    send({ type: 'error', message });
  }

  res.end();
});

const PORT = Number(process.env.PORT ?? 3001);
app.listen(PORT, () => {
  console.log(`SudChat backend → http://localhost:${PORT}`);
});
