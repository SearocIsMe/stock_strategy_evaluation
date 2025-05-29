# System Architecture Diagrams

## 📊 High-Level System Flow

```mermaid
graph TB
    subgraph "Data Sources"
        A1[Tick Data Stream]
        A2[Order Book Data]
        A3[Market News]
        A4[Corporate Actions]
    end
    
    subgraph "Data Ingestion"
        B1[Data Validators]
        B2[Stream Processors]
        B3[Data Normalizers]
    end
    
    subgraph "Storage Layer"
        C1[(ClickHouse<br/>Time-Series DB)]
        C2[(Redis<br/>Cache)]
        C3[(S3<br/>Model Storage)]
    end
    
    subgraph "Factor Engineering"
        D1[Traditional Factors]
        D2[AI-Generated Factors]
        D3[Factor Evaluation]
    end
    
    subgraph "ML Pipeline"
        E1[CNN Model]
        E2[LSTM Model]
        E3[XGBoost]
        E4[Ensemble]
    end
    
    subgraph "Strategy Engine"
        F1[Signal Generation]
        F2[Risk Management]
        F3[Portfolio Optimization]
    end
    
    subgraph "Output"
        G1[Daily Recommendations]
        G2[Performance Reports]
        G3[Real-time Dashboard]
    end
    
    A1 & A2 & A3 & A4 --> B1
    B1 --> B2 --> B3
    B3 --> C1
    C1 --> D1 & D2
    D1 & D2 --> D3
    D3 --> C1
    C1 --> E1 & E2 & E3
    E1 & E2 & E3 --> E4
    E4 --> F1
    F1 --> F2 --> F3
    F3 --> G1 & G2 & G3
    
    C2 -.-> D1
    C2 -.-> E4
    C3 -.-> E1 & E2 & E3
```

## 🔄 Data Processing Pipeline

```mermaid
sequenceDiagram
    participant Market as Market Data
    participant Ingestion as Data Ingestion
    participant DB as ClickHouse
    participant Factor as Factor Engine
    participant ML as ML Models
    participant Strategy as Strategy Engine
    participant Output as Output System
    
    Market->>Ingestion: Tick/Order Book Stream
    Ingestion->>Ingestion: Validate & Normalize
    Ingestion->>DB: Store Raw Data
    
    DB->>Factor: Fetch Historical Data
    Factor->>Factor: Calculate Traditional Factors
    Factor->>Factor: Generate AI Factors (LLM)
    Factor->>DB: Store Factor Values
    
    DB->>ML: Load Features
    ML->>ML: Train/Predict
    ML->>Strategy: Predictions
    
    Strategy->>Strategy: Rank Stocks
    Strategy->>Strategy: Apply Risk Rules
    Strategy->>Output: Generate Signals
    
    Output->>Output: Create Reports
    Output->>Output: Update Dashboard
```

## 🧠 Machine Learning Architecture

```mermaid
graph LR
    subgraph "Feature Engineering"
        A1[Tick Data]
        A2[Order Book]
        A3[Factors]
        A4[Feature Selection]
    end
    
    subgraph "Deep Learning Models"
        B1[CNN<br/>Pattern Recognition]
        B2[LSTM<br/>Sequence Modeling]
        B3[Transformer<br/>Attention Mechanism]
    end
    
    subgraph "Tree Models"
        C1[XGBoost<br/>Gradient Boosting]
        C2[LightGBM<br/>Fast Training]
        C3[Random Forest<br/>Robustness]
    end
    
    subgraph "Ensemble"
        D1[Weighted Average]
        D2[Stacking]
        D3[Meta-Learning]
    end
    
    subgraph "Output"
        E1[Probability Scores]
        E2[Feature Importance]
        E3[Model Confidence]
    end
    
    A1 & A2 & A3 --> A4
    A4 --> B1 & B2 & B3
    A4 --> C1 & C2 & C3
    B1 & B2 & B3 --> D1
    C1 & C2 & C3 --> D1
    D1 --> D2 --> D3
    D3 --> E1 & E2 & E3
```

## 📈 Factor Generation Process

```mermaid
flowchart TD
    A[Market Data] --> B{Data Type}
    B -->|Tick Data| C[Microstructure Factors]
    B -->|Order Book| D[Liquidity Factors]
    B -->|Price/Volume| E[Technical Factors]
    
    C --> F[Traditional Factor Pool]
    D --> F
    E --> F
    
    A --> G[LLM Analysis]
    G --> H[Market Context]
    H --> I[AI Factor Generator]
    
    F --> J[Factor Evaluation]
    I --> J
    
    J --> K{IC > Threshold?}
    K -->|Yes| L[Active Factor Set]
    K -->|No| M[Discard]
    
    L --> N[Factor Storage]
    
    subgraph "LLM Process"
        G --> O[Pattern Detection]
        O --> P[Factor Hypothesis]
        P --> Q[Formula Generation]
        Q --> I
    end
```

## 🎯 Strategy Execution Flow

```mermaid
stateDiagram-v2
    [*] --> DataCollection: Market Open
    
    DataCollection --> FactorCalculation: Data Ready
    FactorCalculation --> Prediction: Factors Computed
    
    Prediction --> Ranking: ML Inference Done
    Ranking --> RiskCheck: Top N Selected
    
    RiskCheck --> PositionSizing: Risk Approved
    RiskCheck --> Rejected: Risk Exceeded
    
    PositionSizing --> SignalGeneration: Sizes Calculated
    SignalGeneration --> Execution: Signals Ready
    
    Execution --> Monitoring: Orders Placed
    Monitoring --> Performance: Track Results
    
    Performance --> [*]: Market Close
    Rejected --> [*]: No Trade
```

## 🏗️ Infrastructure Components

```mermaid
graph TB
    subgraph "Application Layer"
        A1[Strategy Service<br/>Python]
        A2[Factor Service<br/>Python]
        A3[Model Service<br/>Python + GPU]
        A4[API Gateway<br/>FastAPI]
    end
    
    subgraph "Data Layer"
        B1[(ClickHouse Cluster<br/>3 Nodes)]
        B2[(Redis Cluster<br/>Cache)]
        B3[S3 Compatible<br/>Object Storage]
    end
    
    subgraph "Infrastructure"
        C1[Docker Containers]
        C2[Kubernetes<br/>Orchestration]
        C3[Prometheus<br/>Monitoring]
        C4[Grafana<br/>Visualization]
    end
    
    subgraph "External"
        D1[Market Data API]
        D2[LLM API<br/>Qwen/GPT]
        D3[Notification<br/>Services]
    end
    
    A1 & A2 & A3 --> A4
    A4 --> B1 & B2
    A3 --> B3
    
    A1 & A2 & A3 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    
    D1 --> A1
    D2 --> A2
    A1 --> D3
```

## 🔐 Security Architecture

```mermaid
graph LR
    subgraph "External"
        A1[Market Data]
        A2[User Access]
    end
    
    subgraph "Security Layer"
        B1[TLS/SSL]
        B2[API Keys]
        B3[OAuth 2.0]
        B4[Rate Limiting]
    end
    
    subgraph "Application"
        C1[Input Validation]
        C2[Authentication]
        C3[Authorization]
        C4[Audit Logging]
    end
    
    subgraph "Data"
        D1[Encryption at Rest]
        D2[Access Control]
        D3[Data Masking]
    end
    
    A1 --> B1 --> C1
    A2 --> B2 & B3 --> C2 --> C3
    B4 --> C1
    
    C1 & C2 & C3 --> C4
    C4 --> D1 & D2 & D3
```

## 📊 Performance Monitoring

```mermaid
graph TD
    subgraph "Metrics Collection"
        A1[System Metrics<br/>CPU, Memory, Disk]
        A2[Application Metrics<br/>Latency, Throughput]
        A3[Business Metrics<br/>P&L, Sharpe, Drawdown]
    end
    
    subgraph "Processing"
        B1[Prometheus<br/>Time-Series DB]
        B2[Alert Manager]
        B3[Log Aggregation<br/>ELK Stack]
    end
    
    subgraph "Visualization"
        C1[Grafana Dashboards]
        C2[Custom Dash App]
        C3[Email Reports]
    end
    
    subgraph "Actions"
        D1[Auto-Scaling]
        D2[Alert Notifications]
        D3[Performance Tuning]
    end
    
    A1 & A2 & A3 --> B1
    B1 --> B2
    A1 & A2 & A3 --> B3
    
    B1 --> C1
    B1 & B3 --> C2
    B2 --> C3
    
    B2 --> D1 & D2
    C1 & C2 --> D3
```

## 🔄 Backup and Recovery Flow

```mermaid
sequenceDiagram
    participant Scheduler as Backup Scheduler
    participant Script as Backup Script
    participant DB as ClickHouse
    participant Storage as S3/Cloud Storage
    participant Monitor as Monitoring
    
    Scheduler->>Script: Trigger Daily Backup
    Script->>DB: Export Data (Native Format)
    DB-->>Script: Data Dump
    Script->>Script: Compress Data
    Script->>Storage: Upload to Cloud
    Storage-->>Script: Confirmation
    Script->>Monitor: Log Success
    
    Note over Script,Storage: Retention Policy Applied
    
    alt Disaster Recovery
        Monitor->>Script: Initiate Recovery
        Script->>Storage: Download Backup
        Storage-->>Script: Backup Data
        Script->>Script: Decompress
        Script->>DB: Restore Data
        DB-->>Monitor: Recovery Complete
    end
```

## 🚀 Deployment Pipeline

```mermaid
gitGraph
    commit id: "Feature Development"
    branch feature
    checkout feature
    commit id: "Add New Factor"
    commit id: "Update Model"
    checkout main
    merge feature
    commit id: "Run Tests"
    branch staging
    checkout staging
    commit id: "Deploy to Staging"
    commit id: "Integration Tests"
    checkout main
    merge staging
    commit id: "Tag Release"
    branch production
    checkout production
    commit id: "Deploy to Production"
    commit id: "Monitor Performance"
```

## 📈 Daily Strategy Workflow

```mermaid
gantt
    title Daily Strategy Execution Timeline
    dateFormat HH:mm
    axisFormat %H:%M
    
    section Pre-Market
    Data Preparation     :08:00, 30m
    Factor Calculation   :08:30, 30m
    Model Inference      :09:00, 20m
    Signal Generation    :09:20, 10m
    
    section Trading Hours
    Market Open          :milestone, 09:30, 0m
    Position Entry       :09:30, 30m
    Real-time Monitoring :10:00, 5h
    
    section Post-Market
    Market Close         :milestone, 15:00, 0m
    Performance Analysis :15:00, 30m
    Report Generation    :15:30, 30m
    Model Retraining     :16:00, 2h
```

## 🎯 System Integration Points

```mermaid
graph TB
    subgraph "Core System"
        A[Strategy Engine]
    end
    
    subgraph "Data Providers"
        B1[Exchange APIs]
        B2[Data Vendors]
        B3[News Feeds]
    end
    
    subgraph "Execution"
        C1[Broker API]
        C2[Order Management]
        C3[Risk Controls]
    end
    
    subgraph "Analytics"
        D1[Performance Analytics]
        D2[Risk Analytics]
        D3[Factor Analytics]
    end
    
    subgraph "External Services"
        E1[LLM APIs]
        E2[Cloud Storage]
        E3[Notification Services]
    end
    
    B1 & B2 & B3 --> A
    A --> C1 --> C2 --> C3
    A --> D1 & D2 & D3
    A <--> E1 & E2
    A --> E3
```

These diagrams provide a comprehensive visual representation of the system architecture, data flows, and operational processes. They can be rendered using any Mermaid-compatible viewer or documentation system.