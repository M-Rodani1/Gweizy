import React from 'react';

type SandboxedIframeProps = React.IframeHTMLAttributes<HTMLIFrameElement> & {
  title: string;
  src: string;
  sandbox?: string;
};

const DEFAULT_SANDBOX = 'allow-scripts allow-same-origin';

const SandboxedIframe: React.FC<SandboxedIframeProps> = ({
  sandbox = DEFAULT_SANDBOX,
  ...props
}) => {
  return <iframe sandbox={sandbox} {...props} />;
};

export default SandboxedIframe;
