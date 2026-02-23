import Anthropic from '@anthropic-ai/sdk';

// ── Tool definition (sent to Claude so it knows the tool exists) ──────────────

export const fetchUrlTool: Anthropic.Tool = {
  name: 'fetch_url',
  description:
    'Fetch the text content of a web page so it can be read, summarized, or analyzed. ' +
    'Use this whenever the user asks to summarize, read, or analyze a URL.',
  input_schema: {
    type: 'object' as const,
    properties: {
      url: {
        type: 'string',
        description: 'The full URL to fetch (must start with http:// or https://)',
      },
    },
    required: ['url'],
  },
};

// ── Tool implementation (runs on the server when Claude calls it) ─────────────

export async function fetchUrl(url: string): Promise<string> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10_000);

    const response = await fetch(url, {
      signal: controller.signal,
      headers: { 'User-Agent': 'SudChat/1.0 (URL summarizer bot)' },
    });
    clearTimeout(timeout);

    if (!response.ok) {
      return `Error: HTTP ${response.status} ${response.statusText}`;
    }

    const contentType = response.headers.get('content-type') ?? '';
    const raw = await response.text();

    let content: string;
    if (contentType.includes('text/html')) {
      // Strip scripts, styles, and HTML tags — keep readable text
      content = raw
        .replace(/<script[\s\S]*?<\/script>/gi, '')
        .replace(/<style[\s\S]*?<\/style>/gi, '')
        .replace(/<[^>]+>/g, ' ')
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#039;/g, "'")
        .replace(/&nbsp;/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
    } else {
      content = raw.trim();
    }

    // Limit size to avoid hitting Claude's context window
    if (content.length > 50_000) {
      content = content.slice(0, 50_000) + '\n\n[Content truncated at 50,000 characters]';
    }

    return content || 'No readable text content found on this page.';
  } catch (err: unknown) {
    if (err instanceof Error) {
      if (err.name === 'AbortError') return 'Error: Request timed out after 10 seconds.';
      return `Error fetching URL: ${err.message}`;
    }
    return 'Error: Unknown error while fetching the URL.';
  }
}
