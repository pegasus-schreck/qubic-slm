import requests
from bs4 import BeautifulSoup
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Dict, Tuple, Optional
import os
from tqdm import tqdm
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.corpus import stopwords
from collections import Counter
import logging
from datetime import datetime

nltk.download('averaged_perceptron_tagger_eng')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_nltk():
    """Download and set up required NLTK data."""
    try:
        required_packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
        for package in required_packages:
            nltk.download(package, quiet=True)
        logger.info("NLTK setup completed successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        raise

class CryptoQADataset(Dataset):
    """Enhanced Dataset class for crypto documentation QA pairs."""
    
    def __init__(self, qa_pairs: List[Dict], tokenizer, max_length: int = 512):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        question = qa_pair['question']
        context = qa_pair['context']
        answer_text = qa_pair['answer']
        
        try:
            # Improved tokenization with sliding window
            encoding = self.tokenizer(
                question,
                context,
                max_length=self.max_length,
                truncation='longest_first',
                padding='max_length',
                return_tensors='pt',
                return_offsets_mapping=True,
                stride=128
            )
            # Remove the batch dimension since DataLoader will handle it
            for key in encoding.keys():
                if torch.is_tensor(encoding[key]):
                    encoding[key] = encoding[key].squeeze(0)
            
            # Find answer start and end positions
            start_idx = context.index(answer_text)
            end_idx = start_idx + len(answer_text)

            # Convert to token positions
            offset_mapping = encoding['offset_mapping'].tolist()
            start_token = None
            end_token = None

            for idx, (start, end) in enumerate(offset_mapping):
                if start <= start_idx < end:
                    start_token = idx
                if start < end_idx <= end:
                    end_token = idx
                    break

            # Fallback handling
            if start_token is None or end_token is None:
                start_token = 0
                end_token = 1

            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'start_positions': torch.tensor(start_token),
                'end_positions': torch.tensor(end_token)
            }
            
        except Exception as e:
            logger.warning(f"Error processing QA pair: {str(e)}")
            # Return default values in case of error
            encoding = self.tokenizer(
                question,
                context,
                max_length=self.max_length,
                truncation='longest_first',
                padding='max_length',
                return_tensors='pt'
            )
            
            # Remove batch dimension
            default_encoding = {k: v.squeeze(0) for k, v in default_encoding.items()}
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'start_positions': torch.tensor(0),
                'end_positions': torch.tensor(1)
            }

class CryptoDocsQA:
    """Enhanced Question Answering system for cryptocurrency documentation."""
    
    def __init__(self, base_url: str, output_dir: str = "qubic"):
        self.base_url = base_url
        self.output_dir = output_dir
        self.docs_data = []
        
        # Model initialization with better error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                'bert-large-uncased-whole-word-masking-finetuned-squad'
            )
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                'bert-large-uncased-whole-word-masking-finetuned-squad'
            )
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize feedback storage
        self.feedback_file = os.path.join(output_dir, 'feedback.json')
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                self.feedback_data = json.load(f)
        else:
            self.feedback_data = []

    def scrape_documentation(self) -> None:
        """Enhanced documentation scraping with better error handling and validation."""
        try:
            logger.info(f"Starting documentation scrape from {self.base_url}")
            response = requests.get(self.base_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all documentation links
            doc_links = set()
            for link in soup.find_all('a', href=True):
                href = link['href']

                if self._is_valid_doc_link(self._build_full_url(link['href'])):
                    doc_url = self._build_full_url(link['href'])
                    doc_links.add(doc_url)
                    
                    # Go a layer deeper to grab more details
                    sub_response = requests.get(doc_url, timeout=10)
                    sub_response.raise_for_status()
                    sub_soup = BeautifulSoup(sub_response.text, 'html.parser')
                    
                    sub_links = set()
                    for sub_link in sub_soup.find_all('a', href=True):
                        sub_href = sub_link['href']
                        
                        if self._is_valid_doc_link(self._build_full_url(sub_link['href'])):
                            sub_url = self._build_full_url(sub_link['href'])
                            doc_links.add(sub_url)
                        
            
            logger.info(f"Found {len(doc_links)} documentation pages")
            
            # Process each link with progress bar
            for url in tqdm(doc_links, desc="Scraping documentation"):
                try:
                    logger.info(f"Scraping page {url}")
                    content = self._extract_page_content(url)
                    if content and len(content.strip()) > 100:
                        cleaned_content = self.clean_text(content)
                        if len(cleaned_content.split()) > 20:  # Minimum word count
                            self.docs_data.append({
                                'url': url,
                                'title': self._extract_title(url),
                                'content': cleaned_content,
                                'timestamp': datetime.now().isoformat()
                            })
                except Exception as e:
                    logger.error(f"Error processing {url}: {str(e)}")
                    continue
            
            if self.docs_data:
                self._save_docs_data()
                logger.info(f"Successfully scraped {len(self.docs_data)} documents")
            else:
                raise ValueError("No valid documents were scraped")
                
        except Exception as e:
            logger.error(f"Scraping failed: {str(e)}")
            raise

    def _save_docs_data(self) -> None:
        """Save scraped documentation data to disk."""
        with open(os.path.join(self.output_dir, 'docs_data.json'), 'w') as f:
            json.dump(self.docs_data, f)

    def _extract_title(self, url: str) -> str:
        """Extract page title with improved accuracy."""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try different title elements in order of preference
            title = None
            
            # Check main content area first
            main_content = soup.find('main') or soup.find('article')
            if main_content:
                title = main_content.find('h1')
            
            # Try page title if no h1 found
            if not title:
                title = soup.find('title')
            
            # Clean and return title text
            if title:
                title_text = title.get_text().strip()
                # Remove common suffixes
                title_text = re.sub(r'\s*[|\-]\s*.*$', '', title_text)
                return title_text
            
            return os.path.basename(url)
            
        except Exception as e:
            logger.error(f"Error extracting title from {url}: {str(e)}")
            return url

    def _extract_page_content(self, url: str) -> str:
        """Extract page content with improved cleaning and structuring."""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.select('nav, footer, header, script, style, iframe, .sidebar, .menu'):
                element.decompose()
            
            # Try different content containers
            content_selectors = [
                'main',
                'article',
                '.content',
                '.documentation',
                '#main-content',
                '.main-content',
                'p',
                'li'
            ]
            
            content = None
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break
            
            if not content:
                content = soup.find('body')
            
            if content:
                # Process text with special handling
                text_parts = []
                for element in content.descendants:
                    if element.name is None and element.string:
                        text = element.string.strip()
                        if text:
                            text_parts.append(text)
                    elif element.name in ['code', 'pre']:
                        code = element.get_text().strip()
                        if code:
                            text_parts.append(f"Code example: {code}")
                    elif element.name == 'h1':
                        heading = element.get_text().strip()
                        if heading:
                            text_parts.append(f"\n{heading}\n")
                    elif element.name in ['h2', 'h3', 'h4']:
                        heading = element.get_text().strip()
                        if heading:
                            text_parts.append(f"\n{heading}")
                
                return '\n'.join(text_parts)
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return ""

    def _is_valid_doc_link(self, href: str) -> bool:
        
        """Check if the link is a valid documentation page."""
        # Customize based on the website's URL pattern
        href_lower = href.lower()
        
        skip_links = ['#', 'blog.qubic.org', 'discord', 'twitter', 'copilot', 'github.com/qubic/docs', 'gitlab.com', 'linkednin', 'tradeogre', 'apple.com', 
                      'careers', 'gate.io', 'hibt.com', 'bit2me', 'githubstatus', 'safe.trade', 'bitpanda', 'gethashwallet']
        
        if any(pattern in href_lower for pattern in skip_links):
            return False
        return True
    
    def _build_full_url(self, href: str) -> str:
        """Build full URL with improved handling of different formats."""
        if href.startswith('http'):
            return href
        return f"{self.base_url.rstrip('/')}/{href.lstrip('/')}"
        
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better preservation of technical terms."""
        # Remove HTML
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Preserve technical terms and symbols
        text = re.sub(r'[^\w\s\.\-\(\)\/\[\]\{\}\:\,\;\'\"\`]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common issues
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        
        return text.strip()

    def prepare_context(self, question: str) -> Tuple[str, List[str]]:
        """Enhanced context preparation with semantic search."""
        if not self.docs_data:
            raise ValueError("No documentation data available")
        
        # Create enhanced TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.95,
            min_df=1,
            analyzer='char_wb'  # Better for technical terms
        )
        
        # Prepare document segments
        doc_segments = []
        doc_sources = []
        window_size = 3
        
        for doc in self.docs_data:
            sentences = sent_tokenize(doc['content'])
            for i in range(0, len(sentences), window_size):
                segment = ' '.join(sentences[i:i + window_size])
                doc_segments.append(segment)
                doc_sources.append(doc['url'])
        
        try:
            # Calculate similarities
            X = tfidf.fit_transform(doc_segments)
            q = tfidf.transform([question.lower()])
            
            similarities = cosine_similarity(q, X).squeeze()
            
            # Get top relevant segments
            top_k = 5
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            contexts = []
            sources = []
            
            for idx in top_indices:
                if similarities[idx] > 0.05:
                    contexts.append(doc_segments[idx])
                    sources.append(doc_sources[idx])
            
            combined_context = ' '.join(contexts)
            
            # Smart truncation
            if len(combined_context) > 5000:
                sentences = sent_tokenize(combined_context)
                combined_context = ' '.join(sentences[:25])
            
            return combined_context, list(set(sources))
            
        except Exception as e:
            logger.error(f"Error in prepare_context: {str(e)}")
            return self.docs_data[0]['content'], [self.docs_data[0]['url']]

    def answer_question(self, question: str) -> Dict:
        """Enhanced answer extraction with improved confidence scoring."""
        context, sources = self.prepare
        
    def answer_question(self, question: str) -> Dict:
        """Enhanced answer extraction with improved confidence scoring."""
        context, sources = self.prepare_context(question)
        
        # Tokenize input with sliding window
        inputs = self.tokenizer(
            question,
            context,
            max_length=512,
            truncation='only_second',
            stride=128,
            return_tensors="pt",
            padding=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )
        
        # Process each window
        all_answers = []
        offset_mapping = inputs.pop('offset_mapping')
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if k != 'overflow_to_sample_mapping'}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process each window's predictions
        for i in range(len(outputs.start_logits)):
            start_logits = outputs.start_logits[i]
            end_logits = outputs.end_logits[i]
            
            # Get top-k answer candidates
            top_k = 3
            start_indices = torch.topk(start_logits, top_k).indices
            end_indices = torch.topk(end_logits, top_k).indices
            
            for start_idx in start_indices:
                for end_idx in end_indices:
                    if start_idx <= end_idx:
                        answer_tokens = inputs['input_ids'][i][start_idx:end_idx + 1]
                        answer = self.tokenizer.decode(answer_tokens)
                        
                        if self._is_valid_answer(answer):
                            confidence = float(
                                torch.softmax(start_logits, dim=0)[start_idx] *
                                torch.softmax(end_logits, dim=0)[end_idx]
                            )
                            
                            all_answers.append({
                                'text': answer,
                                'confidence': confidence
                            })
        
        # Select best answer
        if all_answers:
            filtered_answers = [
                a for a in all_answers
                if len(a['text'].split()) >= 3 and
                len(a['text'].split()) <= 50 and
                a['confidence'] > 0.01
            ]
            
            if filtered_answers:
                best_answer = max(filtered_answers, key=lambda x: x['confidence'])
                return {
                    "answer": best_answer['text'].strip(),
                    "confidence": best_answer['confidence'],
                    "sources": sources
                }
        
        return {
            "answer": "I couldn't find a confident answer to this question.",
            "confidence": 0.0,
            "sources": sources
        }

    def _is_valid_answer(self, answer: str) -> bool:
        """Validate answer quality."""
        # Remove special tokens and clean
        answer = answer.strip()
        if not answer:
            return False
            
        # Remove common BERT special tokens
        answer = re.sub(r'\[CLS\]|\[SEP\]', '', answer).strip()
        
        # Check minimum length
        if len(answer.split()) < 3:
            return False
            
        # Check for incomplete sentences
        if answer[-1] not in '.?!':
            return False
            
        return True

    def generate_qa_pairs(self, num_pairs: int = 200) -> List[Dict]:
        """
        Generate diverse question-answer pairs from documentation.
    
        Args:
            num_pairs: Number of QA pairs to generate
        
        Returns:
            List of dictionaries containing questions, answers, and contexts
        """
        qa_pairs = []
    
        # Enhanced question patterns by category
        question_patterns = {
         'definition': [
                "What is {}?",
                "Define {}.",
                "Can you explain what {} is?",
                "What does {} mean?",
                "Describe {}.",
            ],
            'process': [
                "How does {} work?",
                "What is the process of {}?",
                "How is {} implemented?",
                "Explain the mechanism of {}.",
                "What are the steps in {}?",
            ],
           'purpose': [
                "What is the purpose of {}?",
                "Why is {} used?",
                "What are the benefits of {}?",
                "What problem does {} solve?",
                "What is {} used for?",
            ],
            'technical': [
                "What are the technical specifications of {}?",
                "What are the components of {}?",
                "How is {} structured?",
                "What are the requirements for {}?",
                "What are the features of {}?",
            ],
            'comparison': [
                "How does {} differ from other approaches?",
                "What makes {} unique?",
                "What are the advantages of {}?",
                "How does {} compare to alternatives?",
                "What distinguishes {} from others?",
            ]
        }
    
        # Process each document
        for doc in self.docs_data:
            content = doc['content']
            
            # Split into paragraphs and sentences
            paragraphs = [p for p in content.split('\n') if len(p.strip()) > 0]
            
            for para in paragraphs:
                # Split paragraph into sentences
                sentences = sent_tokenize(para)
                if len(sentences) < 2:
                    continue
                
                # Extract technical terms and concepts
                technical_terms = self._extract_technical_terms(para)
                
                for term in technical_terms:
                    # Skip too short or too long terms
                    if len(term.split()) < 2 or len(term.split()) > 5:
                        continue
                    
                    # Determine question type based on context
                    question_type = self._determine_question_type(term, para)
                    
                    # Generate question using appropriate pattern
                    question = random.choice(question_patterns[question_type]).format(term)
                    
                    # Find most relevant sentence as answer
                    answer_sentence = self._find_most_relevant_sentence(term, sentences)
                    
                    if answer_sentence:
                        # Add context from surrounding sentences
                        context_start = max(0, sentences.index(answer_sentence) - 1)
                        context_end = min(len(sentences), sentences.index(answer_sentence) + 2)
                        context = ' '.join(sentences[context_start:context_end])
                        
                        qa_pairs.append({
                            'question': question,
                            'context': context,
                            'answer': answer_sentence,
                            'answer_start': context.index(answer_sentence)
                        })
                        
                        if len(qa_pairs) >= num_pairs:
                            return qa_pairs
    
        return qa_pairs

    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms and key phrases from text."""
        # Tokenize and tag parts of speech
        words = word_tokenize(text.lower())
        tagged = nltk.pos_tag(words)
        
        # Get stop words
        stop_words = set(stopwords.words('english'))
        
        # Technical term patterns
        technical_patterns = [
            ['NN', 'NN'],  # Noun + Noun
            ['JJ', 'NN'],  # Adjective + Noun
            ['NN', 'NN', 'NN'],  # Multiple Nouns
            ['JJ', 'NN', 'NN'],  # Adjective + Multiple Nouns
        ]
        
        terms = []
        i = 0
        while i < len(tagged):
            # Try to match each pattern
            for pattern in technical_patterns:
                if i + len(pattern) <= len(tagged):
                    # Check if sequence matches pattern
                    sequence = tagged[i:i + len(pattern)]
                    if all(tag.startswith(expected) for (word, tag), expected in zip(sequence, pattern)):
                        # Extract words from matching sequence
                        term_words = [word for word, tag in sequence]
                        # Check if words are not stop words
                        if not any(word in stop_words for word in term_words):
                            terms.append(' '.join(term_words))
            i += 1
        
        return list(set(terms))

    def _determine_question_type(self, term: str, context: str) -> str:
        """Determine the most appropriate question type based on context."""
        context_lower = context.lower()
        term_lower = term.lower()
        
        # Check for different types of content
        if any(word in context_lower for word in ['how', 'process', 'implement', 'step', 'procedure']):
            return 'process'
        elif any(word in context_lower for word in ['why', 'purpose', 'benefit', 'advantage', 'use case']):
            return 'purpose'
        elif any(word in context_lower for word in ['component', 'specification', 'require', 'feature']):
            return 'technical'
        elif any(word in context_lower for word in ['compare', 'differ', 'unlike', 'versus', 'better']):
            return 'comparison'
        else:
            return 'definition'

    def _find_most_relevant_sentence(self, term: str, sentences: List[str]) -> Optional[str]:
        """Find the most relevant sentence for a given term."""
        best_sentence = None
        highest_score = -1
        
        term_words = set(term.lower().split())
        
        for sentence in sentences:
            score = self._calculate_sentence_relevance(term, sentence)
            
            if score > highest_score:
                highest_score = score
                best_sentence = sentence
        
        return best_sentence

    def _calculate_sentence_relevance(self, term: str, sentence: str) -> float:
        """Calculate relevance score between term and sentence."""
        term_lower = term.lower()
        sentence_lower = sentence.lower()
        
        # Base score
        score = 0.0
        
        # Exact match gets highest score
        if term_lower in sentence_lower:
            score += 1.0
        
        # Word overlap
        term_words = set(term_lower.split())
        sentence_words = set(sentence_lower.split())
        word_overlap = len(term_words & sentence_words)
        score += 0.3 * (word_overlap / len(term_words))
        
        # Position bonus (earlier sentences preferred)
        if sentence_lower.startswith(term_lower):
            score += 0.2
        
        # Length penalty (prefer shorter sentences)
        length_penalty = min(1.0, 20.0 / len(sentence.split()))
        score *= length_penalty
        
        return score

    def prepare_training_data(self, train_ratio: float = 0.8, max_length: int = 512):
        """Prepare training and evaluation datasets."""
        qa_pairs = self.generate_qa_pairs(num_pairs=200)
        
        # Filter out pairs with too long sequences
        filtered_pairs = []
        for pair in qa_pairs:
            total_length = len(self.tokenizer.encode(pair['question'])) + \
                        len(self.tokenizer.encode(pair['context']))
            if total_length <= max_length:
                filtered_pairs.append(pair)
        
        # Shuffle and split the filtered pairs
        random.shuffle(filtered_pairs)
        split_idx = int(len(filtered_pairs) * train_ratio)
        
        train_pairs = filtered_pairs[:split_idx]
        eval_pairs = filtered_pairs[split_idx:]
        
        logger.info(f"Created {len(train_pairs)} training and {len(eval_pairs)} evaluation samples")
        
        # Create datasets with consistent max_length
        train_dataset = CryptoQADataset(train_pairs, self.tokenizer, max_length=max_length)
        eval_dataset = CryptoQADataset(eval_pairs, self.tokenizer, max_length=max_length)
        
        return train_dataset, eval_dataset
    
    def invoke_training(self):
        self.model.train()
        
    def fine_tune_model(self, train_dataset, eval_dataset, num_epochs=5):
        """Enhanced model fine-tuning with improved training strategy."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        def collate_fn(batch):
            """Custom collate function to ensure consistent tensor sizes."""
            return {
                'input_ids': torch.stack([x['input_ids'] for x in batch]),
                'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
                'start_positions': torch.stack([x['start_positions'] for x in batch]),
                'end_positions': torch.stack([x['end_positions'] for x in batch])
            }    
    
        # Training configuration
        batch_size = 8
        train_loader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  collate_fn=collate_fn)
        eval_loader = DataLoader(eval_dataset, 
                                 batch_size=batch_size,
                                 collate_fn=collate_fn)
        
        # Optimizer with layer-specific learning rates
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if 'bert' in n.lower()],
                'lr': 1e-5
            },
            {
                'params': [p for n, p in self.model.named_parameters() if 'bert' not in n.lower()],
                'lr': 3e-5
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_eval_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Evaluation phase
            self.model.eval()
            total_eval_loss = 0
            
            with torch.no_grad():
                for batch in eval_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    total_eval_loss += outputs.loss.item()
            
            avg_eval_loss = total_eval_loss / len(eval_loader)
            
            logger.info(f"Epoch {epoch+1}:")
            logger.info(f"  Average training loss: {avg_train_loss:.4f}")
            logger.info(f"  Average evaluation loss: {avg_eval_loss:.4f}")
            
            # Save best model
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                model_path = os.path.join(self.output_dir, 'best_model')
                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break

    def save_feedback(self, question: str, answer: str, helpful: bool):
        """Save user feedback for future improvements."""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'helpful': helpful
        }
        self.feedback_data.append(feedback)
        
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)

def main():
    """Main execution function."""
    setup_nltk()
    
    qa_system = CryptoDocsQA(base_url="https://docs.qubic.org/")
    
    try:
        # Load or scrape data
        data_file = os.path.join(qa_system.output_dir, 'docs_data.json')
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                qa_system.docs_data = json.load(f)
            logger.info(f"Loaded {len(qa_system.docs_data)} documents")
        else:
            logger.info("Scraping documentation...")
            qa_system.scrape_documentation()
        
        # Prepare training data
        logger.info("Preparing training data...")
        train_dataset, eval_dataset = qa_system.prepare_training_data()
        
        # Fine-tune model
        logger.info("Fine-tuning model...")
        qa_system.fine_tune_model(train_dataset, eval_dataset)
        
        # Interactive QA
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ").strip()
            if question.lower() == 'quit':
                break
            
            if len(question) < 5:
                print("Please enter a more specific question.")
                continue
            
            result = qa_system.answer_question(question)
            
            print("\nAnswer:", result["answer"])
            print(f"Confidence: {result['confidence']:.2%}")
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")
            
            # Collect feedback
            feedback = input("\nWas this answer helpful? (y/n): ").lower()
            qa_system.save_feedback(question, result["answer"], feedback == 'y')
            qa_system.invoke_training()
    
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
