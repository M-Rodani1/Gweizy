const DEFAULT_API_PROXY_TARGET = 'https://basegasfeesml-production.up.railway.app';

function getProxyTarget(env: { API_PROXY_TARGET?: string }): string {
  const configured = env.API_PROXY_TARGET?.trim();
  if (!configured) {
    return DEFAULT_API_PROXY_TARGET;
  }
  return configured.replace(/\/+$/, '');
}

function buildTargetUrl(requestUrl: string, pathParam: string | string[] | undefined, baseTarget: string): string {
  const incoming = new URL(requestUrl);
  const pathStr = Array.isArray(pathParam) ? pathParam.join('/') : (pathParam ?? '');
  const suffix = pathStr.length > 0 ? `/${pathStr}` : '';
  return `${baseTarget}/api${suffix}${incoming.search}`;
}

export async function onRequest(context: {
  request: Request;
  params: { path?: string | string[] };
  env: { API_PROXY_TARGET?: string };
}): Promise<Response> {
  const { request, params, env } = context;
  const targetBase = getProxyTarget(env);
  const targetUrl = buildTargetUrl(request.url, params.path, targetBase);

  const headers = new Headers(request.headers);
  headers.delete('host');
  headers.set('x-forwarded-host', new URL(request.url).host);
  headers.set('x-forwarded-proto', new URL(request.url).protocol.replace(':', ''));

  const init: RequestInit = {
    method: request.method,
    headers,
    redirect: 'follow'
  };

  if (request.method !== 'GET' && request.method !== 'HEAD') {
    init.body = request.body;
  }

  try {
    return await fetch(targetUrl, init);
  } catch (error) {
    return new Response(
      JSON.stringify({
        success: false,
        error: 'API proxy unavailable'
      }),
      {
        status: 502,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-store'
        }
      }
    );
  }
}
