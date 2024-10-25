import numpy as np
from gensim.models import AuthorTopicModel
from scipy.special import gamma, gammaln
import time


# Helper functions for the Beta-Liouville Distribution
def beta_liouville_pdf(x, alpha, beta):
    """
    Beta-Liouville Probability Density Function
    """
    normalization = np.exp(
        gammaln(np.sum(alpha + beta)) - np.sum(gammaln(alpha + beta))
    )
    prod_term = np.prod(
        [x[i] ** (alpha[i] - 1) * (1 - x[i]) ** (beta[i] - 1) for i in range(len(x))]
    )
    return normalization * prod_term


class AuthorTopicModel:
    def __init__(
        self, corpus=None, num_topics=10, id2word=None, author2doc=None, *args, **kwargs
    ):
        self.corpus = corpus
        self.num_topics = num_topics
        self.id2word = id2word
        self.author2doc = author2doc


class ABLIMA(AuthorTopicModel):
    def __init__(
        self,
        corpus=None,
        num_topics=10,
        id2word=None,
        authors=None,
        alpha_init=0.1,
        beta_init=0.1,
        a_init=0.1,
        b_init=0.1,
        *args,
        **kwargs,
    ):
        self.vocab_size = len(id2word)

        # Beta-Liouville specific parameters
        self.alpha = np.full(num_topics, alpha_init)
        self.beta = np.full(num_topics, beta_init)
        self.a = np.full(num_topics, a_init)
        self.b = np.full(num_topics, b_init)

        self.authors = list(authors.keys())

        super(ABLIMA, self).__init__(
            corpus=corpus,
            num_topics=num_topics,
            id2word=id2word,
            author2doc=authors,
            *args,
            **kwargs,
        )

        # Count matrices and topic assignments
        self.word_topic_matrix = np.zeros((num_topics, len(self.id2word)))
        self.author_topic_matrix = np.zeros((len(self.authors), num_topics))
        self.topic_counts = np.zeros(num_topics)
        self.topic_assignments = []

        # Initialize topic assignments randomly
        for doc_id, document in enumerate(corpus):
            current_doc_assignments = []
            for word_pos, (word_id, _) in enumerate(document):
                initial_topic = np.random.choice(num_topics)
                current_doc_assignments.append(initial_topic)
                self.word_topic_matrix[initial_topic][word_id] += 1
                self.topic_counts[initial_topic] += 1
                author = self.get_author(doc_id)
                author_idx = self.authors.index(author)
                self.author_topic_matrix[author_idx][initial_topic] += 1
            self.topic_assignments.append(current_doc_assignments)

    def get_author(self, doc_id):
        return next(
            author for author, docs in self.author2doc.items() if doc_id in docs
        )

    def get_current_topic(self, doc_id, word_pos):
        return self.topic_assignments[doc_id][word_pos]

    def decrement_counts(self, doc_id, word_pos, current_topic, author):
        word_id = self.corpus[doc_id][word_pos][0]
        self.word_topic_matrix[current_topic][word_id] -= 1
        self.topic_counts[current_topic] -= 1
        author_idx = self.authors.index(author)
        self.author_topic_matrix[author_idx][current_topic] -= 1

    def calculate_phi_update(self):
        return (self.word_topic_matrix + self.b[:, np.newaxis]) / (
            self.word_topic_matrix.sum(axis=1)[:, np.newaxis]
            + self.vocab_size * self.b[:, np.newaxis]
        )

    def calculate_theta_update(self, author_idx):
        """
        Update the author-topic distribution using Beta-Liouville.
        """
        author_topic_totals = self.author_topic_matrix[author_idx]
        theta = (author_topic_totals + self.a) / (
            np.sum(author_topic_totals) + np.sum(self.a)
        )

        # Apply the Beta-Liouville PDF normalization
        normalization = beta_liouville_pdf(theta, self.alpha, self.beta)

        # Apply normalization to ensure consistency with the distribution
        theta_normalized = theta / normalization
        return theta_normalized

    def calculate_topic_probabilities(self, doc_id, word_id, author):
        author_idx = self.authors.index(author)
        author_probs = self.calculate_theta_update(author_idx)
        word_probs = self.calculate_phi_update()[:, word_id]
        combined_probs = author_probs * word_probs

        if np.any(combined_probs < 0):
            raise ValueError(f"Negative probabilities detected: {combined_probs}")

        # Normalization step
        normalized_probs = combined_probs / np.sum(combined_probs)

        if np.any(normalized_probs < 0):
            raise ValueError(
                f"Normalization resulted in negative probabilities: {normalized_probs}"
            )

        # Ensure probabilities sum to 1
        if not np.isclose(np.sum(normalized_probs), 1):
            raise ValueError(f"Probabilities do not sum to 1: {normalized_probs}")

        return normalized_probs

    def gibbs_sampling(self, iterations=10):
        for iteration in range(iterations):
            print(f"iteration: {iteration+1}")

            # Record the current time
            start_time = time.time()
            for doc_id, document in enumerate(self.corpus):
                author = self.get_author(doc_id)
                author_idx = self.authors.index(author)

                for word_pos, (word_id, _) in enumerate(document):
                    current_topic = self.get_current_topic(doc_id, word_pos)
                    self.decrement_counts(doc_id, word_pos, current_topic, author)

                    # Calculate topic probabilities
                    topic_probs = self.calculate_topic_probabilities(
                        doc_id, word_id, author
                    )
                    new_topic = np.random.choice(self.num_topics, p=topic_probs)

                    self.topic_assignments[doc_id][word_pos] = new_topic

                    # Update matrices
                    self.word_topic_matrix[new_topic, word_id] += 1
                    self.topic_counts[new_topic] += 1
                    self.author_topic_matrix[author_idx, new_topic] += 1

            # Record the current time after the iteration has completed
            end_time = time.time()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(f"Time : {elapsed_time:.4f} seconds \n")
