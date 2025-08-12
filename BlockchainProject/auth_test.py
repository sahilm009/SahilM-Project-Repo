import os
from google.cloud import bigquery
import pandas as pd


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'replace with your path i used to have mine here'

def test_bigquery_connection(project_id="replace with your project id i used to have mine here"):
    
    
    print("Testing BigQuery connection...")
    
    try:
        # Create client
        client = bigquery.Client(project=project_id)
        print(f"Client created for project: {project_id}")
        
        # Test basic query - very small and free
        test_query = """
        SELECT 
            from_address,
            block_timestamp,
            value
        FROM `bigquery-public-data.crypto_ethereum.transactions`
        WHERE DATE(block_timestamp) = '2024-07-01'
        LIMIT 10
        """
        
        print("Running small test query...")
        query_job = client.query(test_query)
        results = query_job.to_dataframe()
        
        print(f"Success retrieved {len(results)} test transactions")
        print(f"Sample data from: {results['block_timestamp'].min()}")
        print(f"Sample transaction values: {results['value'].head().tolist()}")
        
        return True, results
        
    except Exception as e:
        print(f"Connection failed: {e}")
        
        # Check common issues
        if "403" in str(e):
            print("Issue: Permissions error - check if service account has BigQuery User role")
        elif "401" in str(e):
            print("Issue: Authentication error - check JSON key file path")
        elif "404" in str(e):
            print("Issue: Project not found - check project ID")
        else:
            print("General BigQuery setup issue")
            
        return False, None

def run_small_ethereum_query(project_id="blockchaindata-468620"):
    
    
    try:
        client = bigquery.Client(project=project_id)
        
       
        schema_query = """
        SELECT column_name, data_type
        FROM `bigquery-public-data.crypto_ethereum.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = 'transactions'
        ORDER BY ordinal_position
        """
        
        print("Checking available columns...")
        schema_job = client.query(schema_query)
        columns = schema_job.to_dataframe()
        
        gas_columns = columns[columns['column_name'].str.contains('gas', case=False, na=False)]
        print("Available gas-related columns:")
        print(gas_columns[['column_name', 'data_type']].to_string())
        
        
        query = """
        SELECT 
            from_address as wallet_id,
            `hash` as tx_hash,
            block_timestamp as timestamp,
            CAST(value AS FLOAT64) / 1e18 as eth_value
        FROM `bigquery-public-data.crypto_ethereum.transactions`
        WHERE 
            DATE(block_timestamp) = '2024-07-01'
            AND value > 0
        LIMIT 1000
        """
        
        print("Running small Ethereum query (1000 transactions, basic columns)...")
        query_job = client.query(query)
        df = query_job.to_dataframe()
        
        
        df['value_usd'] = df['eth_value'] * 3000  # Approximate ETH price
        
        print(f"✅ SUCCESS! Retrieved {len(df)} real Ethereum transactions")
        print(f"🏦 Unique wallets: {df['wallet_id'].nunique()}")
        print(f"💰 Total volume: ${df['value_usd'].sum():,.2f}")
        print(f"📊 Column names in result: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Query failed: {e}")
        return None

if __name__ == "__main__":
    print("BIGQUERY AUTHENTICATION TEST")
    print("=" * 40)
    
   
    success, test_data = test_bigquery_connection()
    
    if success:
        print("\n" + "=" * 40)
        print("Running small Ethereum query...")
        
        
        eth_data = run_small_ethereum_query()
        
        if eth_data is not None:
            print("\nAll tests passed")
        else:
            print("\n Small query failed")
    else:
        print("\n Authentication failed")