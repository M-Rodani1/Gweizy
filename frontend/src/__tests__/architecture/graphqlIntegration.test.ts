import { describe, it, expect, vi } from 'vitest';
import { createGraphQLClient } from '../../utils/graphqlClient';

describe('graphql integration prep', () => {
  it('handles successful graphql responses', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch' as any).mockResolvedValue({
      json: async () => ({ data: { status: 'ok' } })
    } as any);

    const client = createGraphQLClient('https://example.com/graphql');
    const result = await client<{ status: string }>('query { status }');

    expect(result.status).toBe('ok');
    fetchSpy.mockRestore();
  });

  it('throws on graphql errors', async () => {
    vi.spyOn(globalThis, 'fetch' as any).mockResolvedValue({
      json: async () => ({ errors: [{ message: 'Bad request' }] })
    } as any);

    const client = createGraphQLClient('https://example.com/graphql');
    await expect(client('query { status }')).rejects.toThrow('Bad request');
  });
});
