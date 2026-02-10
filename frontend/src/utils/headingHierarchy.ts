export interface HeadingHierarchyIssue {
  message: string;
  fromLevel?: number;
  toLevel?: number;
  headingText?: string | null;
}

export interface HeadingHierarchyResult {
  isValid: boolean;
  issues: HeadingHierarchyIssue[];
}

const HEADING_SELECTOR = 'h1, h2, h3, h4, h5, h6';

const getHeadingLevel = (heading: Element): number => {
  const level = Number(heading.tagName.slice(1));
  return Number.isNaN(level) ? 0 : level;
};

export const validateHeadingHierarchy = (container: ParentNode): HeadingHierarchyResult => {
  const headings = Array.from(container.querySelectorAll(HEADING_SELECTOR));
  const issues: HeadingHierarchyIssue[] = [];
  let previousLevel: number | null = null;

  headings.forEach((heading) => {
    const level = getHeadingLevel(heading);
    if (!level) return;

    if (previousLevel !== null && level - previousLevel > 1) {
      issues.push({
        message: `Heading level skipped: h${previousLevel} -> h${level}`,
        fromLevel: previousLevel,
        toLevel: level,
        headingText: heading.textContent?.trim() ?? null,
      });
    }

    previousLevel = level;
  });

  return {
    isValid: issues.length === 0,
    issues,
  };
};
