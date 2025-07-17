import os
import glob
import json
import time
import logging
import requests
import pandas as pd


class CoreDB:
    def __init__(self, root_dir: str):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.api_endpoint = "https://api.core.ac.uk/v3/search/works"
        self.root = root_dir
        self.get_api_key()
    
    def get_api_key(self):
        with open(f"{self.root}/apikey.txt", "r") as file:
            self.api_key = file.readlines()[0].strip()
    
    def query_api(self, query, is_scroll = True, limit = 100, scrollId = None):
        headers={"Authorization": f"Bearer {self.api_key}"}
        query = {"q": query, "limit": limit}
        if not is_scroll:
            response = requests.post(f"{self.api_endpoint}", data=json.dumps(query), headers=headers)
        elif not scrollId:
            query["scroll"] = "true"
            response = requests.post(f"{self.api_endpoint}", data=json.dumps(query), headers=headers)
        else:
            query["scrollId"] = scrollId
            response = requests.post(f"{self.api_endpoint}", data=json.dumps(query), headers=headers)
        if response.status_code == 200:
            return response.json(), response.elapsed.total_seconds()
        else:
            logging.error(f"Error code {response.status_code}, {response.content}")
            return response.status_code, response.content
    
    def scroll(self, query: str, ceiling: int = 1000, i: int = 0):
        allresults = []
        scrollId = None
        while True:
            bool_ = True
            while bool_:
                result, elapsed = self.query_api(query, is_scroll=True, scrollId=scrollId)
                if result == 500:
                    time.sleep(2)
                elif result == 429:
                    logging.info('Retrying in 60 seconds...')
                    time.sleep(60)
                else:
                    bool_ = False
            scrollId = result["scrollId"]
            totalhits = result["totalHits"]
            result_size = len(result["results"])
            if result_size == 0:
                break
            for hit in result["results"]:
                if self.paper_metadata_check(hit):
                    allresults.append(self.hit_to_doc(hit))
            count = len(allresults)
            logging.info(f"{count}/{totalhits} {elapsed}s")
            if count >= ceiling:
                break
        
        try:
            output_path = f"{self.root}/data/metadata_{i}.parquet"
            pd.DataFrame(allresults).to_parquet(output_path, index=False)
            logging.info(f"Metadata saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving metadata to parquet: {e}")
    
    def concat_metadata(self):
        parquet_files = sorted(glob.glob(f"{self.root}/data/metadata_*.parquet"))
        if not parquet_files:
            logging.warning("No metadata parquet files found to concatenate.")
            return

        df_list = [pd.read_parquet(f) for f in parquet_files]
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.drop_duplicates(subset=['title'], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        output_path = f"{self.root}/data/metadata.parquet"
        combined_df.to_parquet(output_path, index=False)
        logging.info(f"Combined metadata saved to {output_path}")

        for f in parquet_files:
            try:
                os.remove(f)
                logging.info(f"Deleted {f}")
            except Exception as e:
                logging.error(f"Failed to delete {f}: {e}")

    def hit_to_doc(self, hit):
        full_text = hit.get("fullText", "")
        ft_path = f"{self.root}/data/full_texts/{hit['id']}.txt"
        with open(ft_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        row = {
            'id': hit.get("id", ""),
            'title': hit.get("title", ""),
            'abstract': hit.get("abstract", ""),
            'documentType': hit.get("documentType", ""),
            'authors': hit.get("authors", []),
            'yearPublished': hit.get("yearPublished", ""),
            'fullText': ft_path,
            'downloadUrl': hit.get("downloadUrl", ""),
            'language': hit.get("language", "")
        }
        return row

    def paper_metadata_check(self, response):
        # Check for required fields
        required_fields = ["fullText", "abstract", "title"]
        for field in required_fields:
            if field not in response or not response[field]:
                return False

        # Check if full_text is legible English (basic check)
        full_text = response["fullText"]
        # Heuristic: at least 80% of characters are printable and contains spaces
        printable_chars = sum(c.isprintable() for c in full_text)
        space_count = full_text.count(' ')
        if printable_chars / max(len(full_text), 1) < 0.8 or space_count < 10:
            return False

        # Heuristic: check for presence of common English words
        common_words = ["the", "and", "of", "in", "to", "with", "for"]
        if not any(word in full_text.lower() for word in common_words):
            return False

        return True