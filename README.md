# RFS Chatbot: Recipes for Science Knowledge Assistant

A ground-up implementation of a Retrieval-Augmented Generation (RAG) chatbot without relying on any heavyweight frameworks. This chatbot is specifically designed to answer questions about the "Recipes for Science" textbook.

## Overview

This project demonstrates how to build a simple but effective RAG-based conversational AI system. It uses:

* OpenAI's embedding and completion models
* ChromaDB for vector storage and similarity search
* Custom document chunking and retrieval logic

The chatbot retrieves relevant passages from the "Recipes for Science" textbook and uses them as context for generating accurate, contextually relevant responses to user questions about philosophy of science concepts.
## Features

* Semantic Search: Uses embeddings to find relevant textbook passages
* Context-Aware Responses: Leverages GPT-4o to generate accurate answers based on retrieved context
* Evaluation Framework: Includes tools to measure faithfulness, relevancy, and correctness of responses
* Simple CLI Interface: Easy-to-use command line interaction

## Installation

**Clone** the repository:

```bash 
git clone https://github.com/yourusername/rfs-chatbot.git
cd rfs-chatbot
```

Create a **virtual environment**:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install **dependencies**:

```bash
pip install -r requirements.txt
```
Set up **environment variables**:

```bash
cp .env.example .env
# Edit .env to add your OpenAI API key
```
## Usage

### Running the Chatbot

```bash
python src/app.py
```

- Load the configuration
- Check if the collection already exists in ChromaDB
- If not, create it by processing and embedding the textbook
- Start an interactive chat session

### Evaluation

To evaluate the chatbot's performance:
```bash
python tests/eval.py
```

This will run a series of test questions and evaluate the responses for:

* Faithfulness to the source material
* Relevancy to the question
* Correctness compared to expected answers

### Configuration

The chatbot's behavior can be customized through the config/config.yaml file:

## **Future Improvements**

* Web interface for easier interaction
* Multi-document support for expanded knowledge base
* Conversation history and context management
* Response citation and source tracking
* Streaming responses for better UX