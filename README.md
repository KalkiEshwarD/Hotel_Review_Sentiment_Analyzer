# Hotel Feedback Sentiment and Service Extractor

This project aims to develop a machine learning program that analyzes hotel feedback to determine overall sentiment and extract specific information regarding services mentioned.

## Table of Contents
- [Introduction](#introduction)
- [Objective](#objective)
- [Tools and Technologies](#tools-and-technologies)
- [Methodology](#methodology)
- [Output Results](#output-results)
- [Challenges Faced](#challenges-faced)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [License](#license)

## Introduction

Businesses today are inundated with vast amounts of **unstructured text data**, particularly in the form of customer feedback. For industries like hospitality, manually sifting through countless reviews and comments to glean meaningful insights is a time-consuming and often inefficient process. This challenge frequently leads to missed opportunities for service improvement and a delayed response to guest concerns. This project introduces a **machine learning program** designed to meet this demand, offering a systematic approach to understanding and utilizing hotel guest feedback more effectively.

## Objective

The objective of this project is to develop a robust machine learning program capable of **automating the comprehensive analysis of hotel guest feedback**. Specifically, the program aims to achieve the following: first, to accurately determine the **overall sentiment** (positive, negative, or neutral) expressed within each feedback submission. Second, it will identify and extract all **specific hotel services** mentioned by guests within their feedback. Third, for each identified service, the program will assess and assign an **individual sentiment**, indicating the guest's specific feeling towards that particular service. Finally, the program will extract and present **relevant descriptive information** or details that clarify the nature of the feedback pertaining to each service. This structured output will enable hotels to gain deeper, more granular insights into guest experiences, facilitating targeted operational enhancements and improved guest satisfaction.

## Tools and Technologies

The development and execution of this project relied on the following key tools and technologies:

* **Visual Studio Code:** Served as the primary Integrated Development Environment (IDE) for writing, debugging, and managing the project's Python codebase.
    * Reference: [https://code.visualstudio.com/](https://code.visualstudio.com/)
* **LM Studio (Local LLM Server):** Utilized as a local inference server to host and manage the large language model, enabling offline and controlled API access for model querying.
    * Reference: [https://lmstudio.ai/](https://lmstudio.ai/)
* **Google’s Gemma-3-12b open-source LLM:** The specific large language model employed for sentiment analysis and information extraction tasks, hosted locally via LM Studio.
    * Reference: [https://blog.google/technology/ai/gemma-open-models/](https://blog.google/technology/ai/gemma-open-models/)
* **Python:** The core programming language used for developing the entire application logic, including data processing, API interactions, and output generation.
    * Reference: [https://www.python.org/](https://www.python.org/)
* **Onlyoffice Docspace Software:** Used for editing and managing the CSV files, serving as both the input for raw feedback and the output for structured analysis results.
    * Reference: [https://www.onlyoffice.com/docspace/](https://www.onlyoffice.com/docspace/)
* **GitHub:** Employed for version control and hosting the project's codebase, facilitating development and allowing for tracking changes.
    * Reference: [https://github.com/](https://github.com/)
* **Google Gemini 2.5 Flash:** Aided as an AI assistant during various stages of the project, including code generation, debugging, and content refinement for documentation.
    * Reference: [https://ai.google/models/gemini/](https://ai.google/models/gemini/)

## Methodology

This project employs a hybrid approach to hotel review analysis, combining keyword-based heuristics with a Large Language Model (LLM) hosted via LM Studio. The entire process is orchestrated by a Python program (`main.py`) designed to ingest customer feedback, process it systematically, and output structured analytical results.

The methodology can be broken down into the following key stages:

1.  **Data Ingestion:**
    * The program begins by reading hotel review data from a specified CSV file (e.g., `hotel_feedback.csv`) using the `pandas` library.
    * It intelligently identifies the column containing the review text by looking for common headers such as 'review', 'feedback', 'comment', or 'text'. If no explicit header is found, it defaults to the first available text-based column.

2.  **Overall Sentiment Analysis:**
    * For each review, the program first determines the overall sentiment. This is primarily achieved by sending the review text to the configured LM Studio endpoint with a specific prompt. The LLM is instructed to classify the sentiment as "Positive," "Negative," or "Neutral."
    * A robust fallback mechanism is implemented: if the LLM API call fails or returns an unclassifiable response, a keyword-based heuristic is employed. This fallback counts occurrences of predefined positive and negative keywords within the review to infer the overall sentiment, ensuring continuity of analysis even in the event of LLM unavailability.

3.  **Service Identification:**
    * The program identifies relevant hotel services mentioned within each review. This is done by checking the review text against a predefined dictionary of service categories (e.g., "Room Quality," "Dining," "Service") and their associated keywords.
    * To maintain focus and conciseness in the output, a maximum of two most relevant service categories are selected per review for further detailed analysis.

4.  **Service-Specific Sentiment and Feedback Extraction:**
    * For each identified relevant service, a more granular analysis is performed. A targeted prompt is constructed and sent to the LM Studio LLM, requesting:
        * The sentiment specifically related to that service (Positive, Negative, or Neutral).
        * The exact snippet or portion of the review that talks about this service, limited to a concise length (e.g., max 50 words) to capture specific feedback.
    * Similar to the overall sentiment, a fallback mechanism is in place for service-specific analysis. If the LLM call fails or its response cannot be parsed, the system reverts to a keyword-based approach. It identifies sentences containing service-specific keywords and then applies the general keyword-based sentiment analysis to these isolated sentences to infer service sentiment and extract the relevant text.

5.  **Output Generation:**
    * All extracted information—including the original review number, overall sentiment, identified services, their respective sentiments, and specific feedback excerpts—is compiled into a structured DataFrame.
    * This structured data is then saved to a new CSV file (e.g., `review_analysis.csv`), making the analysis results easily accessible for further examination or integration into other systems.
    * A summary of the total processed reviews and the distribution of overall sentiments is also printed to the console upon completion.

Throughout the process, error handling and logging are integrated to monitor the execution, manage API call failures through retry logic with exponential backoff, and provide informative messages for debugging and operational oversight. A small time delay is also introduced between API calls to prevent overwhelming the LM Studio server.

## Output Results

**Input format:**
Sl No, Feedback

**Output format:**
Review number, Overall sentiment, Service 1, Sentiment 1, Review 1, Service 2, Sentiment 2, Review 2,...
