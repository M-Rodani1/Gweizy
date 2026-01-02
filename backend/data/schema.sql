-- Gas Prices Table
CREATE TABLE IF NOT EXISTS gas_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    current_gas REAL NOT NULL,
    base_fee REAL NOT NULL,
    priority_fee REAL NOT NULL,
    block_number INTEGER NOT NULL
);

-- Predictions Table
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    horizon VARCHAR(10) NOT NULL,
    predicted_gas REAL NOT NULL,
    actual_gas REAL,
    model_version VARCHAR(50)
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_gas_prices_timestamp ON gas_prices(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_horizon ON predictions(horizon);

