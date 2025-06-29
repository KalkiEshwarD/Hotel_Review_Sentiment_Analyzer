import pandas as pd
import requests
import json
import re
from typing import Dict, List, Tuple, Optional
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HotelReviewAnalyzer:
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1/chat/completions"):
        """
        Initialize the Hotel Review Analyzer
        
        Args:
            lm_studio_url: URL of your LM Studio server API endpoint
        """
        self.lm_studio_url = lm_studio_url
        self.services = {
            "Room Quality": ["room", "bed", "bathroom", "clean", "comfort", "amenities", "towel", "shower", "ac", "air conditioning", "temperature"],
            "Service": ["staff", "employee", "reception", "concierge", "helpful", "friendly", "professional", "rude", "service"],
            "Dining": ["food", "restaurant", "breakfast", "dinner", "lunch", "meal", "kitchen", "room service", "menu", "taste"],
            "Facilities": ["pool", "spa", "gym", "fitness", "sauna", "jacuzzi", "recreation", "activities", "wifi", "internet"],
            "Value": ["price", "cost", "expensive", "cheap", "worth", "value", "money", "budget", "affordable"],
            "Check-in/Check-out": ["check-in", "check-out", "checkout", "checkin", "arrival", "departure", "registration", "key"]
        }
        
    def call_lm_studio(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Call LM Studio API with retry logic
        
        Args:
            prompt: The prompt to send to the model
            max_retries: Maximum number of retry attempts
            
        Returns:
            Model response or None if failed
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "google/gemma-3-12b",  # This can be adjusted based on your LM Studio setup
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # Low temperature for consistent results
            "max_tokens": 500
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.lm_studio_url, headers=headers, json=data, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    logger.warning(f"API call failed with status {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error("All API call attempts failed")
        return None
    
    def extract_overall_sentiment(self, review_text: str) -> str:
        """
        Extract overall sentiment from review using LM Studio
        
        Args:
            review_text: The review text to analyze
            
        Returns:
            Sentiment label (Positive, Negative, or Neutral)
        """
        prompt = f"""
        Analyze the overall sentiment of this hotel review and classify it as exactly one of: Positive, Negative, or Neutral.

        Review: "{review_text}"

        Respond with only one word: Positive, Negative, or Neutral.
        """
        
        response = self.call_lm_studio(prompt)
        if response:
            sentiment = response.strip().title()
            if sentiment in ["Positive", "Negative", "Neutral"]:
                return sentiment
        
        # Fallback to keyword-based analysis
        return self.fallback_sentiment_analysis(review_text)
    
    def fallback_sentiment_analysis(self, text: str) -> str:
        """
        Fallback sentiment analysis using keyword matching
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment label
        """
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "perfect", "love", "best", "fantastic", "outstanding"]
        negative_words = ["bad", "terrible", "awful", "horrible", "worst", "hate", "disgusting", "disappointing", "poor", "unacceptable"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "Positive"
        elif neg_count > pos_count:
            return "Negative"
        else:
            return "Neutral"
    
    def identify_relevant_services(self, review_text: str) -> List[str]:
        """
        Identify which services are mentioned in the review
        
        Args:
            review_text: The review text to analyze
            
        Returns:
            List of relevant service categories
        """
        relevant_services = []
        text_lower = review_text.lower()
        
        for service, keywords in self.services.items():
            if any(keyword in text_lower for keyword in keywords):
                relevant_services.append(service)
        
        return relevant_services[:2]  # Limit to top 2 services to match output format
    
    def extract_service_sentiment(self, review_text: str, service: str) -> Tuple[str, str]:
        """
        Extract sentiment and specific feedback for a service
        
        Args:
            review_text: The review text
            service: The service category to analyze
            
        Returns:
            Tuple of (sentiment, specific_feedback)
        """
        prompt = f"""
        Analyze this hotel review specifically for mentions of "{service}".

        Review: "{review_text}"

        1. What is the sentiment regarding {service}? (Positive, Negative, or Neutral)
        2. Extract the specific part of the review that talks about {service} (max 50 words)

        Format your response as:
        Sentiment: [Positive/Negative/Neutral]
        Feedback: [specific feedback about {service}]
        """
        
        response = self.call_lm_studio(prompt)
        if response:
            try:
                lines = response.strip().split('\n')
                sentiment_line = next((line for line in lines if line.startswith('Sentiment:')), '')
                feedback_line = next((line for line in lines if line.startswith('Feedback:')), '')
                
                sentiment = sentiment_line.replace('Sentiment:', '').strip().title()
                feedback = feedback_line.replace('Feedback:', '').strip()
                
                if sentiment not in ["Positive", "Negative", "Neutral"]:
                    sentiment = "Neutral"
                
                return sentiment, feedback
            except Exception as e:
                logger.warning(f"Error parsing service sentiment response: {e}")
        
        # Fallback to keyword-based analysis
        return self.fallback_service_analysis(review_text, service)
    
    def fallback_service_analysis(self, review_text: str, service: str) -> Tuple[str, str]:
        """
        Fallback service-specific analysis
        
        Args:
            review_text: The review text
            service: The service category
            
        Returns:
            Tuple of (sentiment, feedback)
        """
        keywords = self.services.get(service, [])
        text_lower = review_text.lower()
        
        # Find sentences containing service keywords
        sentences = re.split(r'[.!?]+', review_text)
        relevant_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                relevant_sentences.append(sentence.strip())
        
        feedback = ' '.join(relevant_sentences[:2])  # First 2 relevant sentences
        if len(feedback) > 100:
            feedback = feedback[:97] + "..."
        
        sentiment = self.fallback_sentiment_analysis(feedback) if feedback else "Neutral"
        
        return sentiment, feedback if feedback else f"No specific mention of {service}"
    
    def process_reviews(self, input_file: str, output_file: str):
        """
        Process all reviews from input CSV and generate analysis output
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
        """
        try:
            # Read input CSV
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} reviews from {input_file}")
            
            # Detect review column (common column names)
            review_column = None
            possible_columns = ['review', 'feedback', 'comment', 'text', 'Review', 'Feedback', 'Comment', 'Text']
            
            for col in possible_columns:
                if col in df.columns:
                    review_column = col
                    break
            
            if review_column is None:
                # If not found, use the first text column
                text_columns = df.select_dtypes(include=['object']).columns
                if len(text_columns) > 0:
                    review_column = text_columns[0]
                else:
                    raise ValueError("No suitable review column found in CSV")
            
            logger.info(f"Using column '{review_column}' as review text")
            
            # Prepare output data
            output_data = []
            
            for index, row in df.iterrows():
                review_text = str(row[review_column])
                if pd.isna(review_text) or review_text.strip() == "":
                    continue
                
                logger.info(f"Processing review {index + 1}/{len(df)}")
                
                # Extract overall sentiment
                overall_sentiment = self.extract_overall_sentiment(review_text)
                
                # Identify relevant services
                relevant_services = self.identify_relevant_services(review_text)
                
                # Prepare row data
                row_data = {
                    'Review_Number': index + 1,
                    'Overall_Sentiment': overall_sentiment
                }
                
                # Extract sentiment for each relevant service (up to 2)
                for i, service in enumerate(relevant_services[:2], 1):
                    service_sentiment, service_feedback = self.extract_service_sentiment(review_text, service)
                    
                    # Clean service name for column naming
                    service_clean = service.replace(' ', '_').replace('/', '_')
                    
                    row_data[f'Service_{i}'] = service
                    row_data[f'Sentiment_{i}'] = service_sentiment
                    row_data[f'Review_of_{i}'] = service_feedback
                
                # Fill empty services if less than 2 found
                for i in range(len(relevant_services) + 1, 3):
                    row_data[f'Service_{i}'] = ""
                    row_data[f'Sentiment_{i}'] = ""
                    row_data[f'Review_of_{i}'] = ""
                
                output_data.append(row_data)
                
                # Add small delay to avoid overwhelming the API
                time.sleep(0.5)
            
            # Create output DataFrame
            output_df = pd.DataFrame(output_data)
            
            # Save to CSV
            output_df.to_csv(output_file, index=False)
            logger.info(f"Analysis complete. Results saved to {output_file}")
            
            # Print summary
            print(f"\n=== Analysis Summary ===")
            print(f"Total reviews processed: {len(output_data)}")
            print(f"Sentiment distribution:")
            sentiment_counts = output_df['Overall_Sentiment'].value_counts()
            for sentiment, count in sentiment_counts.items():
                print(f"  {sentiment}: {count}")
            
            return output_df
            
        except Exception as e:
            logger.error(f"Error processing reviews: {str(e)}")
            raise

def main():
    """
    Main function to run the hotel review analysis
    """
    # Configuration
    INPUT_FILE = "hotel_feedback.csv"
    OUTPUT_FILE = "review_analysis.csv"
    LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"  # Adjust if needed
    
    # Initialize analyzer
    analyzer = HotelReviewAnalyzer(lm_studio_url=LM_STUDIO_URL)
    
    try:
        # Process reviews
        results = analyzer.process_reviews(INPUT_FILE, OUTPUT_FILE)
        print(f"\nProcessing completed successfully!")
        print(f"Input file: {INPUT_FILE}")
        print(f"Output file: {OUTPUT_FILE}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{INPUT_FILE}'")
        print("Please make sure the file exists in the current directory.")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to LM Studio at {LM_STUDIO_URL}")
        print("Please make sure LM Studio is running and the URL is correct.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()