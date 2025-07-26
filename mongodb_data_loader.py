#!/usr/bin/env python3
"""
E-commerce Chatbot Data Loader
Loads CSV files into MongoDB collections for the customer support chatbot.

Usage:
    python load_data.py
    
Requirements:
    pip install pymongo pandas
"""

import os
import sys
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, BulkWriteError
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class EcommerceDataLoader:
    """Handles loading CSV data into MongoDB for the e-commerce chatbot."""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017", 
                 database_name: str = "ecommerce_chatbot"):
        """
        Initialize the data loader.
        
        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database to use
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
        
        # Define CSV file to collection mapping
        self.file_collection_mapping = {
            'products.csv': 'products',
            'orders.csv': 'orders',
            'inventory.csv': 'inventory',
            'customers.csv': 'customers',
            'order_items.csv': 'order_items',
            'categories.csv': 'categories'
        }
    
    def connect_to_mongodb(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to MongoDB at {self.connection_string}")
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            
            # Test the connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            
            logger.info(f"Successfully connected to database: {self.database_name}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            return False
    
    def close_connection(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def read_csv_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Read CSV file into pandas DataFrame.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame or None if file couldn't be read
        """
        try:
            logger.info(f"Reading CSV file: {file_path}")
            
            # Read CSV with flexible parsing options
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                na_values=['', 'NULL', 'null', 'N/A', 'n/a'],
                keep_default_na=True,
                dtype=str  # Read everything as string initially
            )
            
            logger.info(f"Successfully read {len(df)} rows from {file_path}")
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except pd.errors.EmptyDataError:
            logger.warning(f"Empty CSV file: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return None
    
    def preprocess_dataframe(self, df: pd.DataFrame, collection_name: str) -> pd.DataFrame:
        """
        Preprocess DataFrame before inserting into MongoDB.
        
        Args:
            df: Input DataFrame
            collection_name: Target collection name
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Preprocessing data for collection: {collection_name}")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean column names (remove spaces, special characters)
        processed_df.columns = processed_df.columns.str.strip().str.replace(' ', '_').str.lower()
        
        # Collection-specific preprocessing
        if collection_name == 'products':
            # Convert price columns to float
            price_columns = ['price', 'cost', 'retail_price']
            for col in price_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        elif collection_name == 'orders':
            # Convert date columns
            date_columns = ['order_date', 'shipped_date', 'delivered_date']
            for col in date_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
            
            # Convert order total to float
            if 'order_total' in processed_df.columns:
                processed_df['order_total'] = pd.to_numeric(processed_df['order_total'], errors='coerce')
        
        elif collection_name == 'inventory':
            # Convert stock quantities to integer
            quantity_columns = ['quantity_in_stock', 'quantity', 'stock_level']
            for col in quantity_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0).astype(int)
        
        elif collection_name == 'order_items':
            # Convert quantity and price columns
            if 'quantity' in processed_df.columns:
                processed_df['quantity'] = pd.to_numeric(processed_df['quantity'], errors='coerce').fillna(0).astype(int)
            if 'unit_price' in processed_df.columns:
                processed_df['unit_price'] = pd.to_numeric(processed_df['unit_price'], errors='coerce')
        
        # Replace NaN values with None for MongoDB
        processed_df = processed_df.where(pd.notnull(processed_df), None)
        
        return processed_df
    
    def insert_data_to_collection(self, df: pd.DataFrame, collection_name: str) -> bool:
        """
        Insert DataFrame data into MongoDB collection.
        
        Args:
            df: DataFrame to insert
            collection_name: Target collection name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            collection = self.db[collection_name]
            
            # Convert DataFrame to list of dictionaries
            documents = df.to_dict('records')
            
            logger.info(f"Inserting {len(documents)} documents into collection: {collection_name}")
            
            # Clear existing data (optional - comment out if you want to append)
            logger.info(f"Clearing existing data from collection: {collection_name}")
            collection.delete_many({})
            
            # Insert documents in batches for better performance
            batch_size = 1000
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result = collection.insert_many(batch, ordered=False)
                logger.info(f"Inserted batch {i//batch_size + 1}: {len(result.inserted_ids)} documents")
            
            # Create useful indexes
            self.create_indexes(collection_name)
            
            logger.info(f"Successfully loaded {len(documents)} documents into {collection_name}")
            return True
            
        except BulkWriteError as e:
            logger.error(f"Bulk write error for collection {collection_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inserting data into collection {collection_name}: {e}")
            return False
    
    def create_indexes(self, collection_name: str):
        """
        Create useful indexes for each collection.
        
        Args:
            collection_name: Name of the collection
        """
        collection = self.db[collection_name]
        
        try:
            if collection_name == 'products':
                collection.create_index("product_id")
                collection.create_index("category")
                collection.create_index("price")
                
            elif collection_name == 'orders':
                collection.create_index("order_id")
                collection.create_index("customer_id")
                collection.create_index("order_date")
                
            elif collection_name == 'inventory':
                collection.create_index("product_id")
                collection.create_index("quantity_in_stock")
                
            elif collection_name == 'customers':
                collection.create_index("customer_id")
                collection.create_index("email")
                
            elif collection_name == 'order_items':
                collection.create_index("order_id")
                collection.create_index("product_id")
            
            logger.info(f"Created indexes for collection: {collection_name}")
            
        except Exception as e:
            logger.warning(f"Error creating indexes for {collection_name}: {e}")
    
    def load_csv_file(self, file_path: str) -> bool:
        """
        Load a single CSV file into MongoDB.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Extract filename from path
        filename = os.path.basename(file_path)
        
        # Check if we have a mapping for this file
        if filename not in self.file_collection_mapping:
            logger.warning(f"No collection mapping found for file: {filename}")
            return False
        
        collection_name = self.file_collection_mapping[filename]
        
        # Read the CSV file
        df = self.read_csv_file(file_path)
        if df is None:
            return False
        
        # Preprocess the data
        processed_df = self.preprocess_dataframe(df, collection_name)
        
        # Insert into MongoDB
        return self.insert_data_to_collection(processed_df, collection_name)
    
    def load_all_csv_files(self, directory: str = ".") -> Dict[str, bool]:
        """
        Load all CSV files from a directory into MongoDB.
        
        Args:
            directory: Directory containing CSV files (default: current directory)
            
        Returns:
            Dict mapping filenames to success status
        """
        results = {}
        
        logger.info(f"Looking for CSV files in directory: {directory}")
        
        # Find all CSV files in the directory
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        
        if not csv_files:
            logger.warning("No CSV files found in the directory")
            return results
        
        logger.info(f"Found {len(csv_files)} CSV files: {csv_files}")
        
        # Process each CSV file
        for filename in csv_files:
            file_path = os.path.join(directory, filename)
            logger.info(f"Processing file: {filename}")
            
            success = self.load_csv_file(file_path)
            results[filename] = success
            
            if success:
                logger.info(f"✅ Successfully loaded: {filename}")
            else:
                logger.error(f"❌ Failed to load: {filename}")
        
        return results
    
    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get document count for each collection.
        
        Returns:
            Dict mapping collection names to document counts
        """
        stats = {}
        
        try:
            collection_names = self.db.list_collection_names()
            
            for collection_name in collection_names:
                count = self.db[collection_name].count_documents({})
                stats[collection_name] = count
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}


def main():
    """Main function to run the data loader."""
    logger.info("Starting E-commerce Chatbot Data Loader")
    
    # Initialize the data loader
    loader = EcommerceDataLoader()
    
    try:
        # Connect to MongoDB
        if not loader.connect_to_mongodb():
            logger.error("Failed to connect to MongoDB. Exiting.")
            sys.exit(1)
        
        # Load all CSV files from current directory
        results = loader.load_all_csv_files()
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("LOADING SUMMARY")
        logger.info("="*50)
        
        successful_files = [f for f, success in results.items() if success]
        failed_files = [f for f, success in results.items() if not success]
        
        logger.info(f"Successfully loaded: {len(successful_files)} files")
        for filename in successful_files:
            logger.info(f"  ✅ {filename}")
        
        if failed_files:
            logger.info(f"Failed to load: {len(failed_files)} files")
            for filename in failed_files:
                logger.info(f"  ❌ {filename}")
        
        # Show collection statistics
        stats = loader.get_collection_stats()
        if stats:
            logger.info("\nCOLLECTION STATISTICS")
            logger.info("-" * 30)
            for collection, count in stats.items():
                logger.info(f"  {collection}: {count:,} documents")
        
        logger.info("\nData loading completed successfully! 🎉")
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Always close the connection
        loader.close_connection()


if __name__ == "__main__":
    main()
