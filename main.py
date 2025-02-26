import sys
from information_retrieval import InformationRetrieval


"""
This is the main file that will be used to run the project. It will take in the following command line arguments:
1. Google API Key
2. Google Engine ID
3. Precision
4. Query
e.g. python main.py <google api key> <google engine id> <precision> <query>
"""
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python main.py <google api key> <google engine id> <precision> <query>")
        sys.exit(1)
    # Get the command line arguments
    API_KEY = sys.argv[1]
    ENGINE_ID = sys.argv[2]
    try:
        PRECISION = float(sys.argv[3])
    except ValueError:
        print("Precision must be a real number")
        sys.exit(1)
    query = sys.argv[4]
    
    # Initialize the InformationRetrieval class
    ir = InformationRetrieval(API_KEY, ENGINE_ID)
    first_iter = True
    
    while True:
        # Display the parameters
        print(f"Parameters:\nClient key  = {API_KEY}\nEngine key  = {ENGINE_ID}\nQuery       = {query}\nPrecision   = {PRECISION}")
        # Get the search results from Google
        results = ir.search(query)
        # Counter for relevant documents
        relevant_docs = 0
        # List to store related docs
        feedback_related = []
        # List to store unrelated docs
        feedback_unrelated = []
        
        # Terminate if in the first iteration there are fewer than 10 results
        if first_iter and len(results) < 10:
            break
        first_iter = False

        # Display the search results
        print("Google Search Results:")
        print("=======================")
        for i, result in enumerate(results):
            print(f"Result {i+1}\n[\n URL: {result['url']}\n Title: {result['title']}\n Summary: {result['summary']}\n]")
            # Request feedback from the user
            relevant = input(f"\nRelevant (Y/N)? ")
            while relevant.upper() not in ['Y', 'N']:
                relevant = input(f"Please enter Y or N\nRelevant (Y/N)? ")
            if relevant.upper() == 'Y':
                relevant_docs += 1
                feedback_related.append(result)
            else:
                feedback_unrelated.append(result)

        # Terminate if there are no relevant documents
        if relevant_docs == 0:
            print("No relevant documents found in this iteration. Exit.")
            break

        # Calculate the precision
        precision = relevant_docs / len(results)
        # Print the feedback summary
        print("=======================")
        print(f"FEEDBACK SUMMARY\nQuery {query}\nPrecision {precision:.1f}")
        if precision < PRECISION:
            print(f"Still below the desired precision of {PRECISION}")
            # Update the query
            query, new_query_words = ir.update_query(query, feedback_related, feedback_unrelated)
            print(f"Augmenting by  {' '.join(new_query_words)}")
        else:
            print(f"Desired precision reached, done")
            break
