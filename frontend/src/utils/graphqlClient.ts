import { API_CONFIG } from '../config/api';
import { withTimeout } from './withTimeout';

export interface GraphQLResponse<T> {
  data?: T;
  errors?: Array<{ message: string }>;
}

export const createGraphQLClient = (endpoint: string) => {
  return async function graphqlRequest<T>(query: string, variables?: Record<string, unknown>): Promise<T> {
    const response = await withTimeout(
      fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, variables })
      }),
      API_CONFIG.TIMEOUT,
      'Request timed out: GraphQL'
    );
    if (typeof response.ok === 'boolean' && !response.ok) {
      throw new Error(`GraphQL request failed with status ${response.status}`);
    }

    const payload = (await response.json()) as GraphQLResponse<T>;
    if (payload.errors?.length) {
      throw new Error(payload.errors[0].message);
    }
    if (!payload.data) {
      throw new Error('No data returned from GraphQL');
    }
    return payload.data;
  };
};
