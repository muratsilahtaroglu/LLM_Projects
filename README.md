# NoCodeTune

**NoCodeTune** is a modular system designed to simplify the fine-tuning process for AI models. It provides a robust framework for managing datasets, configuring jobs, and executing fine-tuning tasks with minimal coding. Designed for scalability and ease of use, **NoCodeTune** aims to empower researchers, developers, and data scientists to focus on innovation rather than repetitive setup tasks.

---

## Key Aims

1. **Automate Fine-Tuning**: Enable seamless training workflows with a no-code interface.
2. **Future-Ready Design**: Lay the foundation for advanced features like data validation, preprocessing, and testing.
3. **Scalability**: Support for multi-GPU and LoRA-based fine-tuning to optimize performance.
4. **Extendable Features**: Create a framework that can grow with the addition of new capabilities.

---

## Current Features

- **Dynamic Job Management**: Create and execute fine-tuning jobs with ease.
- **Flexible Model Support**: Compatible with LLaMA, T5, Pythia, and more.
- **Logging**: Track system events and training progress.
- **Server Integration**: Manage workflows via FastAPI-based endpoints.

---

## Planned Features (Coming Soon)

1. **Data Processing**: Tools for preprocessing and augmenting datasets.
2. **Validation**: Automated checks for data quality and model configurations.
3. **Testing Frameworks**: Model evaluation pipelines to ensure robustness.
4. **Performance Monitoring**: Real-time metrics for training and inference.

---

## Directory Structure

```plaintext
NoCodeTune/
├── FineTuneFlow/            # Dynamic job creation and management.
├── ModelTrainingWorkspace/  # Core workspace for running fine-tuning tasks.
├── archive/                 # Reusable utilities and scripts.
├── files/                   # Shared datasets and input files.
├── jobs/                    # Job configurations and outputs.
├── logs/                    # Logs for debugging and monitoring.
└── scripts/                 # Auxiliary scripts for setup and execution.
```

## Contributors
Developed by: Murat Silahtaroğlu  
Contact: [muratsilahtaroglu13@gmail.com](mailto:muratsilahtaroglu13@gmail.com)