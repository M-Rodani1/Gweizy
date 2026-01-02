import React from 'react';
import {
  ArrowLeftRight,
  BadgeCheck,
  Coins,
  FileCode,
  Image,
  Send,
  Shuffle,
  Sparkles
} from 'lucide-react';
import type { TransactionType } from './chains';

type TxIcon = React.ComponentType<{ className?: string }>;

interface TxMeta {
  label: string;
  shortLabel: string;
  icon: TxIcon;
}

export const TX_TYPE_META: Record<TransactionType, TxMeta> = {
  swap: {
    label: 'Token Swap',
    shortLabel: 'Swap',
    icon: ArrowLeftRight
  },
  bridge: {
    label: 'Bridge',
    shortLabel: 'Bridge',
    icon: Shuffle
  },
  nftMint: {
    label: 'NFT Mint',
    shortLabel: 'Mint',
    icon: Sparkles
  },
  transfer: {
    label: 'ETH Transfer',
    shortLabel: 'Transfer',
    icon: Send
  },
  erc20Transfer: {
    label: 'Token Transfer',
    shortLabel: 'Token Transfer',
    icon: Coins
  },
  approve: {
    label: 'Token Approve',
    shortLabel: 'Approve',
    icon: BadgeCheck
  },
  nftTransfer: {
    label: 'NFT Transfer',
    shortLabel: 'NFT Transfer',
    icon: Image
  },
  contractDeploy: {
    label: 'Deploy Contract',
    shortLabel: 'Deploy',
    icon: FileCode
  }
};

export const getTxLabel = (type: TransactionType): string => {
  return TX_TYPE_META[type]?.label || type;
};

export const getTxShortLabel = (type: TransactionType): string => {
  return TX_TYPE_META[type]?.shortLabel || type;
};
