from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class InformationRetrieval:
    def __init__(self, api_key, engine_id):
        """
        Initialize the InformationRetrieval class

        Args:
            api_key (str): Google API key
            engine_id (str): Google Custom Search Engine ID
        """
        self.engine_id = engine_id
        self.service = build("customsearch", "v1", developerKey=api_key)

        self.stop_words = set()
        with open('stopwords.txt', 'r') as f:
            for line in f:
                self.stop_words.add(line.strip())

    def search(self, user_query):
        """
        Perform Google Custom Search and return results

        Args:
            user_query (str): User query

        Returns:
            list: List of dictionaries containing search results
        """
        try:
            # Execute the search
            result = self.service.cse().list(
                q=user_query,
                cx=self.engine_id,
            ).execute()

            # Format the results
            formatted_results = []
            for item in result['items']:
                # Ignore non-html files
                if 'fileFormat' in item:
                    continue
                formatted_results.append({
                    'url': item['link'],
                    'title': item['title'],
                    'summary': item.get('snippet', '')
                })

            return formatted_results

        except Exception as e:
            print(f"Error performing search: {str(e)}")
            return []

    def update_query(self, user_query, relevant_docs, non_relevant_docs):
        """
        Update the search query based on user feedback

        Args:
            user_query (str): User query
            relevant_docs (list): List of relevant documents
            non_relevant_docs (list): List of non-relevant documents

        Returns:
            string: Updated user query
            list: List of new query terms (at most 2)
        """
        documents = []
        # Remove stopwords
        for doc in relevant_docs:
            words = doc['summary'] + doc['title'] * 2
            processed_words = [word.lower() for word in words.split()
                               if word.lower() not in self.stop_words]
            documents.append(" ".join(processed_words))

        vectorizer = TfidfVectorizer()

        # Calculate TF-IDF for documents and titles
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        # Aggregate TF-IDF scores over documents
        tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)

        # Remove original query terms from candidates
        query_terms = user_query.lower().split()
        for term in query_terms:
            if term in feature_names:
                idx = np.where(feature_names == term)[0][0]
                tfidf_scores[idx] = 0

        # Get indices of terms sorted by TF-IDF scores
        sorted_indices = np.argsort(tfidf_scores)[::-1]

        # Always take the highest scoring term
        expansion_terms = [feature_names[sorted_indices[0]]]

        # Add second term only if it passes threshold
        threshold = 1.5 * np.mean(tfidf_scores)
        if tfidf_scores[sorted_indices[1]] > threshold:
            expansion_terms.append(feature_names[sorted_indices[1]])

        if expansion_terms:
            user_query += ' ' + ' '.join(expansion_terms)

        return user_query, expansion_terms
