from typing import Dict, List
from database_manager import DatabaseManager
from ai_manager import AIManager
import re
import requests
import urllib.parse
import pandas as pd
from collections import defaultdict
from configs import Configs
class RealEstateManager:
    def __init__(self):
        # Initialize database manager
        self.db_manager = DatabaseManager(db_config={
            "host": Configs.DB_HOST,
            "port": Configs.DB_PORT,
            "database": Configs.DB_NAME,
            "user": Configs.DB_USER,
            "password": Configs.DB_PASSWORD})    
        
        # Initialize AI manager with necessary context
        self.ai_manager = AIManager(
            search_constraints=self.db_manager.get_search_constraints(),
            db_schema=self.db_manager.get_database_schema()
        )
        
        # Initialize conversation memory
        self.conversation_memory = []
        self.user_preferences = {
                'listing_type': None,
                'project_status': None,
                'unit_types': None,
                'city': None,
                'neighborhood': None,
                'price': None,
                'rooms': None,
                'area': None,
                'near_by': None,
                'user_location': None,
                'user_coordinates': None,
                'closest_projects_ids': [],
                # 'nearby_item_ids': [],
        }


    def extract_all_gmaps_coordinates(self,user_input) -> List[Dict[str, str]]:
        """Extract all Google Maps coordinates from user input.
        Returns a list of dictionaries with original URL, final resolved URL, latitude, and longitude.
        """
        gmaps_url_pattern = r"(https?://(?:maps\.app\.goo\.gl|goo\.gl/maps|www\.google\.com/maps)[^\s]*)"
        urls = re.findall(gmaps_url_pattern, user_input)

        results = []

        for url in urls:
            final_url = url
            print(f"Processing URL: {url}")

            # Resolve short link
            if "maps.app.goo.gl" in final_url or "goo.gl/maps" in final_url:
                try:
                    resp = requests.head(final_url, allow_redirects=True, timeout=5)
                    final_url = resp.url
                except requests.RequestException:
                    pass

            # Decode URL to handle %2C and similar
            decoded_url = urllib.parse.unquote(final_url)

            # Updated regex to match either "," or "%2C"
            coords_pattern = r"(-?\d+\.\d+)[,|%2C]\s*(?:\+)?(-?\d+\.\d+)"
            coords_match = re.search(coords_pattern, decoded_url)

            if coords_match:
                lat = float(coords_match.group(1))
                lng = float(coords_match.group(2))
                results.append({"original_url": url, "final_url": final_url, "lat": lat, "lng": lng})
            else:
                results.append({"original_url": url, "final_url": final_url, "lat": None, "lng": None})

        return results

    def process_user_message(self, user_message: str,user_preferences:Dict,conversation_memory) -> str:
        """
        Process a user message and return an appropriate response
        """
        self.user_preferences = user_preferences
        # Extract all Google Maps coordinates from user input
        gmaps_coordinates = self.extract_all_gmaps_coordinates(user_message)
        if gmaps_coordinates:
            # Update user preferences with coordinates
            self.user_preferences['user_coordinates'] = (gmaps_coordinates[-1]['lat'], gmaps_coordinates[-1]['lng'])
            self.user_preferences['user_location'] = gmaps_coordinates[-1]['final_url']
            print(f"gmaps_coordinates: {gmaps_coordinates}")
            ids = self.db_manager.get_closest_k_units(
                gmaps_coordinates[-1]['lat'],
                gmaps_coordinates[-1]['lng'],
                radius_km=5,  # Example radius
                top_k=20  # Example top K results
            )
            if ids:
                self.user_preferences['closest_projects_ids'] = [item['project_id'] for item in ids]
            
        
        # Get AI's interpretation of user intent and generate SQL query
        print(f"Processing user message: {user_message}")
        print(f"Current user preferences: {self.user_preferences}")
        self.conversation_memory = conversation_memory
        ai_response = self.ai_manager.intent_detection(
            user_message=user_message,
            user_preferences=self.user_preferences,
            memory=self.conversation_memory
        )

        # Update user preferences based on AI's understanding while preserving other fields
        self.user_preferences.update(ai_response['updated_preferences'])

        # If there's a SQL query, execute it and process results
        if ai_response['sql_query']:
            sql_query = ai_response['sql_query']
            
            # If we have closest_projects_ids, add them to the WHERE clause
            if self.user_preferences['closest_projects_ids']:
                # Check if the query already has a WHERE clause
                if 'WHERE' in sql_query.upper():
                    # Add the project_id condition to existing WHERE clause
                    sql_query = sql_query.replace(
                        'WHERE',
                        f"WHERE project_id IN ({','.join(map(str, self.user_preferences['closest_projects_ids']))}) AND "
                    )
                else:
                    # Add new WHERE clause before any ORDER BY or LIMIT
                    order_by_pos = sql_query.upper().find('ORDER BY')
                    limit_pos = sql_query.upper().find('LIMIT')
                    insert_pos = order_by_pos if order_by_pos != -1 else (limit_pos if limit_pos != -1 else len(sql_query))
                    sql_query = f"{sql_query[:insert_pos]} WHERE project_id IN ({','.join(map(str, self.user_preferences['closest_projects_ids']))}) {sql_query[insert_pos:]}"
            print(f"Executing SQL query: {sql_query}")
            query_results = self.db_manager.execute_sql_query(sql_query)
            # print(f"Query results: {type(query_results)}")
            # print(f"Query results ids: {query_results[0].project_id}")
            if query_results is not None:
                self.ai_manager.query_history.append({
                    'query': ai_response['sql_query'],
                    'results': query_results
                })

                if len(query_results) in range(1, self.ai_manager.MAX_RESULTS_FOR_SUMMARY + 1):
                    project_ids = query_results["project_id"].tolist()
                    nearby_places = self.db_manager.get_nearby_places_for_projects(project_ids)

                    # Group nearby places by project_id
                    places_map = defaultdict(list)
                    for p in nearby_places:
                        places_map[p['project_id']].append({
                            # 'place_id': p['place_id'],
                            'name': p['name'],
                            'type': p['type'],
                            'rating': p['rating'],
                            'address': p['address'],
                            # 'lat': p['lat'],
                            # 'lng': p['lng']
                        })

                    # Merge into final dataframe
                    query_results = pd.DataFrame(query_results)
                    query_results['nearby_places'] = query_results['project_id'].map(places_map)
                    
                    print(query_results)  # or return df if needed
            
            query_results.drop(columns=['coordinates'], errors='ignore', inplace=True)
            
            # Process the results (or lack thereof) and get response
            final_response = self.ai_manager.process_query_results(
                user_query=user_message,
                sql_query=ai_response['sql_query'],
                query_results=query_results,
                user_preferences=self.user_preferences,
                salesman_response = ai_response['salesman_response']
            )
        else:
            # If no SQL query was generated, use the AI's direct response
            final_response = ai_response['salesman_response']

        # Update conversation memory
        self.conversation_memory.append(f"User: {user_message}")
        self.conversation_memory.append(f"Agent: {final_response}")

        return final_response

    def get_current_preferences(self) -> Dict:
        """Return current user preferences"""
        return self.user_preferences

    def reset_preferences(self):
        """Reset user preferences"""
        self.user_preferences = {}

    def get_conversation_history(self) -> List[str]:
        """Return conversation history"""
        return self.conversation_memory
