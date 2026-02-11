import React, { useState, useRef, useEffect, memo } from 'react';

interface LazyImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  src: string;
  alt: string;
  avifSrc?: string;
  webpSrc?: string;
  placeholder?: string;
  className?: string;
}

/**
 * Lazy-loaded image component with intersection observer
 * Only loads image when it enters the viewport
 */
const LazyImage: React.FC<LazyImageProps> = memo(({
  src,
  alt,
  avifSrc,
  webpSrc,
  placeholder = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"%3E%3Crect fill="%231f2937" width="1" height="1"/%3E%3C/svg%3E',
  className = '',
  ...props
}) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isInView, setIsInView] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!imgRef.current) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      {
        rootMargin: '100px',
        threshold: 0.01
      }
    );

    observer.observe(imgRef.current);

    return () => observer.disconnect();
  }, []);

  const image = (
    <img
      ref={imgRef}
      src={isInView ? src : placeholder}
      alt={alt}
      className={`transition-opacity duration-300 ${isLoaded ? 'opacity-100' : 'opacity-0'} ${className}`}
      onLoad={() => setIsLoaded(true)}
      loading="lazy"
      decoding="async"
      {...props}
    />
  );

  if (!avifSrc && !webpSrc) {
    return image;
  }

  return (
    <picture>
      {isInView && avifSrc && <source type="image/avif" srcSet={avifSrc} />}
      {isInView && webpSrc && <source type="image/webp" srcSet={webpSrc} />}
      {image}
    </picture>
  );
});

LazyImage.displayName = 'LazyImage';

export default LazyImage;
