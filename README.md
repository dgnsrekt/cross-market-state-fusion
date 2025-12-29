# Polymarket RL Trading

Reinforcement learning system for trading 15-minute binary prediction markets on Polymarket.

## Overview

This system uses PPO (Proximal Policy Optimization) with MLX for Apple Silicon GPU acceleration to learn trading strategies on Polymarket's 15-minute crypto prediction markets (BTC, ETH, SOL, XRP).

## Architecture

```
├── run.py              # Main trading engine
├── dashboard.py        # Real-time Flask-SocketIO web dashboard
├── strategies/
│   ├── base.py         # Base classes (Action, MarketState, Strategy)
│   ├── rl_mlx.py       # PPO implementation with MLX
│   ├── momentum.py     # Momentum baseline
│   ├── mean_revert.py  # Mean reversion baseline
│   ├── fade_spike.py   # Spike fading baseline
│   └── gating.py       # Ensemble gating strategy
└── helpers/
    ├── polymarket_api.py    # Polymarket REST API
    ├── binance_wss.py       # Binance price streaming
    ├── binance_futures.py   # Futures data (funding, OI, CVD)
    └── orderbook_wss.py     # Polymarket orderbook streaming
```

## Features

- **18-dimensional state space** optimized for 15-min trading:
  - Ultra-short momentum (1m, 5m, 10m returns)
  - Order flow (L1/L5 imbalance, trade flow, CVD acceleration)
  - Microstructure (spread, trade intensity, large trade detection)
  - Volatility regime features
  - Position tracking

- **7-action space** with position sizing:
  - HOLD
  - BUY_SMALL/MEDIUM/LARGE (25%/50%/100%)
  - SELL_SMALL/MEDIUM/LARGE (25%/50%/100%)

- **Dense reward shaping** for sample efficiency:
  - Unrealized PnL delta
  - Transaction cost penalty
  - Spread cost on entry
  - Expiry urgency penalty
  - Momentum alignment bonus

## Usage

### Run with RL strategy (training mode)
```bash
python run.py --strategy rl --train --size 100
```

### Run with RL strategy (inference mode)
```bash
python run.py --strategy rl --load rl_model --size 100
```

### Run dashboard (separate terminal)
```bash
python dashboard.py --port 5001
```

### Other strategies
```bash
python run.py --strategy momentum
python run.py --strategy mean_revert
python run.py --strategy fade_spike
```

## Requirements

- Python 3.11+
- MLX (Apple Silicon)
- websockets
- Flask-SocketIO
- numpy

## Configuration

Key hyperparameters in `strategies/rl_mlx.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_actor` | 3e-4 | Actor learning rate |
| `lr_critic` | 1e-3 | Critic learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_epsilon` | 0.2 | PPO clip range |
| `entropy_coef` | 0.01 | Entropy bonus |
| `buffer_size` | 2048 | Experience buffer size |
| `batch_size` | 128 | Mini-batch size |
| `n_epochs` | 10 | PPO epochs per update |

## Dashboard

Real-time web interface at `http://localhost:5001`:

- Live market probabilities with time remaining
- Position tracking with unrealized P&L
- RL training metrics (policy/value loss, entropy, KL divergence)
- P&L chart and trade history
