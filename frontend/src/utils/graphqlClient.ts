export interface GraphQLResponse<T> {
  data?: T;
  errors?: Array<{ message: string }>;
}

export const createGraphQLClient = (endpoint: string) => {
  return async function graphqlRequest<T>(query: string, variables?: Record<string, unknown>): Promise<T> {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, variables })
    });

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
