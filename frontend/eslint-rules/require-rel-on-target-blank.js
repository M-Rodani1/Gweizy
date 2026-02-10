const isAnchorElement = (node) => node.name && node.name.type === 'JSXIdentifier' && node.name.name === 'a';

const getAttribute = (node, name) =>
  node.attributes.find(
    (attribute) =>
      attribute.type === 'JSXAttribute' &&
      attribute.name &&
      attribute.name.type === 'JSXIdentifier' &&
      attribute.name.name === name
  );

const getStringValue = (attribute) => {
  if (!attribute || !attribute.value) return null;
  if (attribute.value.type === 'Literal') {
    return typeof attribute.value.value === 'string' ? attribute.value.value : null;
  }
  if (
    attribute.value.type === 'JSXExpressionContainer' &&
    attribute.value.expression &&
    attribute.value.expression.type === 'Literal' &&
    typeof attribute.value.expression.value === 'string'
  ) {
    return attribute.value.expression.value;
  }
  return null;
};

export default {
  meta: {
    type: 'problem',
    docs: {
      description: 'Require rel attribute when using target="_blank"',
    },
    schema: [],
    messages: {
      missingRel: 'Links with target="_blank" must include rel="noopener" or rel="noreferrer".',
    },
  },
  create(context) {
    return {
      JSXOpeningElement(node) {
        if (!isAnchorElement(node)) return;

        const targetAttr = getAttribute(node, 'target');
        const targetValue = getStringValue(targetAttr);
        if (targetValue !== '_blank') return;

        const relAttr = getAttribute(node, 'rel');
        const relValue = getStringValue(relAttr);

        if (!relValue) {
          context.report({ node, messageId: 'missingRel' });
          return;
        }

        const normalized = relValue.toLowerCase();
        if (!normalized.includes('noopener') && !normalized.includes('noreferrer')) {
          context.report({ node, messageId: 'missingRel' });
        }
      },
    };
  },
};
