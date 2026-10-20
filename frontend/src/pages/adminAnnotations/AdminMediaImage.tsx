import { useState, type ImgHTMLAttributes } from "react";

export function AdminMediaImage({ src, alt, onError, className, ...props }: ImgHTMLAttributes<HTMLImageElement>) {
  const [failedSrc, setFailedSrc] = useState<string>();
  if (src && failedSrc === src) return <span role="img" aria-label={alt} className={className}>Image indisponible</span>;
  return <img {...props} src={src} alt={alt} className={className} onError={(event) => { setFailedSrc(src); onError?.(event); }} />;
}
