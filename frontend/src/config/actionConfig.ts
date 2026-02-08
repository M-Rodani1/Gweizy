/**
 * Action configuration for TransactionPilot recommendations
 * Defines styling and labels for each AI-recommended action type
 */

export type ActionType = 'WAIT' | 'SUBMIT_NOW' | 'SUBMIT_LOW' | 'SUBMIT_HIGH';

export interface ActionConfig {
  gradient: string;
  cardClass: string;
  text: string;
  subtext: string;
  buttonText: string;
  buttonClass: string;
  confidenceColor: string;
}

const ACTION_CONFIGS: Record<ActionType, Omit<ActionConfig, 'subtext'> & { subtext: string | ((countdown: number | null) => string) }> = {
  WAIT: {
    gradient: 'from-yellow-500 to-orange-500',
    cardClass: 'recommendation-card-yellow',
    text: 'Wait for Better Price',
    subtext: (countdown: number | null) =>
      countdown ? `Optimal in ~${formatCountdown(countdown)}` : 'Prices expected to drop',
    buttonText: 'Notify When Ready',
    buttonClass: 'bg-yellow-500 hover:bg-yellow-600 btn-wait-glow',
    confidenceColor: '#eab308'
  },
  SUBMIT_NOW: {
    gradient: 'from-green-500 to-emerald-500',
    cardClass: 'recommendation-card-green',
    text: 'Execute Now',
    subtext: 'Good time to transact',
    buttonText: 'Execute Transaction',
    buttonClass: 'bg-green-500 hover:bg-green-600 btn-execute-glow',
    confidenceColor: '#22c55e'
  },
  SUBMIT_LOW: {
    gradient: 'from-cyan-500 to-blue-500',
    cardClass: 'recommendation-card-cyan',
    text: 'Try Lower Gas',
    subtext: 'Submit 10% below current (~15% fail risk)',
    buttonText: 'Execute Low',
    buttonClass: 'bg-cyan-500 hover:bg-cyan-600',
    confidenceColor: '#06b6d4'
  },
  SUBMIT_HIGH: {
    gradient: 'from-cyan-600 to-cyan-400',
    cardClass: 'recommendation-card-cyan',
    text: 'Priority Submit',
    subtext: 'Faster confirmation guaranteed',
    buttonText: 'Execute Priority',
    buttonClass: 'bg-cyan-600 hover:bg-cyan-700',
    confidenceColor: '#0891b2'
  }
};

const DEFAULT_CONFIG: ActionConfig = {
  gradient: 'from-gray-500 to-gray-600',
  cardClass: '',
  text: 'Analysing...',
  subtext: 'Getting recommendation',
  buttonText: 'Wait',
  buttonClass: 'bg-gray-500',
  confidenceColor: '#6b7280'
};

function formatCountdown(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function getActionConfig(action: string, countdown?: number | null): ActionConfig {
  const config = ACTION_CONFIGS[action as ActionType];
  if (!config) {
    return DEFAULT_CONFIG;
  }

  return {
    ...config,
    subtext: typeof config.subtext === 'function'
      ? config.subtext(countdown ?? null)
      : config.subtext
  };
}

export function getPrimaryActionLabel(action: string, txShortLabel: string): string {
  switch (action) {
    case 'WAIT':
      return `Notify for ${txShortLabel}`;
    case 'SUBMIT_NOW':
      return `Execute ${txShortLabel}`;
    case 'SUBMIT_LOW':
      return `Execute ${txShortLabel} (Low)`;
    case 'SUBMIT_HIGH':
      return `Priority ${txShortLabel}`;
    default:
      return `Analyse ${txShortLabel}`;
  }
}
