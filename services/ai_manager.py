from typing import Dict, Any, List
import json
from openai import OpenAI
import pandas as pd
from typing import Optional
from configs import Configs
from prompts.prompts import Prompts

class AIManager:
    def __init__(self, search_constraints, db_schema):
        self.openai_client = self.get_openai_client()
        self.search_constraints = search_constraints
        self.db_schema = db_schema
        self.query_history = []
        self.max_results_for_summary = Configs.MAX_RESULTS_FOR_SUMMARY
        self.max_results_for_biolink = Configs.MAX_RESULTS_FOR_BIOLINK
        self.model = Configs.AI_MODEL
        self.temperature = Configs.TEMPERATURE
        self.max_tokens = Configs.MAX_TOKENS
        self.prompts = Prompts()
        

    def update_search_constraints(self, new_constraints: Dict[str, Any]):
        """Update search constraints with new values"""
        self.search_constraints.update(new_constraints)

    def get_openai_client(self):
        """Initialize OpenAI client with API key"""
        api_key = Configs.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return OpenAI(api_key=api_key)

    def intent_detection(self, user_message: str, user_preferences: Dict, memory: List[str]) -> Dict[str, Any]:
        """
        Call OpenAI API to process user message and generate response
        Returns: {
            'sql_query': str or None,
            'salesman_response': str,
            'updated_preferences': dict
        }
        """
        if not self.openai_client:
            return {
                'sql_query': None,
                'salesman_response': "Sorry, I'm unable to process your request right now due to API configuration issues.",
                'updated_preferences': user_preferences.copy()
            }
        ## Drop fields that have none value or empty list
        user_preferences = {k: v for k, v in user_preferences.items() if v not in [None, [], '']}
        system_prompt = self._create_system_prompt(memory, user_preferences)
        print(f"System Prompt: {system_prompt}")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            print(f"AI Response: {response.choices[0].message.content}")
            parsed_response = self._parse_llm_response(response, user_preferences)
            print(f"Parsed Response: {parsed_response}")
            return parsed_response
                
        except Exception as e:
            return {
                'sql_query': None,
                'salesman_response': f"I apologize, but I'm having trouble processing your request right now. Error: {str(e)}",
                'updated_preferences': user_preferences.copy()
            }
    #TODO: Change the LLM Model to 4.1 mini or 5 mini 
    def _create_system_prompt(self, memory: List[str], user_preferences: Dict) -> str:
        """Create the system prompt for the OpenAI API"""
        
        memory_prompt = ' | '.join(memory[-6:]) if memory else 'No previous messages'
        user_preferences_prompt = json.dumps(user_preferences, indent=2, ensure_ascii=False)
        search_constraints_prompt = json.dumps(self.search_constraints, indent=2, ensure_ascii=False)
        return self.prompts.INTENT_DETECTION.format(
            memory=memory_prompt,
            user_preferences=user_preferences_prompt,
            search_constraints=search_constraints_prompt,
            db_schema=self.db_schema
        )        


    def _parse_llm_response(self, response, user_preferences: Dict) -> Dict:
        """Parse the LLM response and handle any errors"""
        try:
            response_text = response.choices[0].message.content
            parsed_response = json.loads(response_text)
            
            if not all(key in parsed_response for key in ['sql_query', 'salesman_response', 'updated_preferences']):
                raise ValueError("Invalid response structure")
                
            return parsed_response
            
        except json.JSONDecodeError:
            return {
                'sql_query': None,
                'salesman_response': response_text,
                'updated_preferences': user_preferences.copy()
            }

    def process_query_results(self, user_query: str, sql_query: str, 
                            query_results: Optional[pd.DataFrame], 
                            user_preferences: Dict,
                            salesman_response:str) -> str:
        """Process query results and generate appropriate response"""
        if query_results is None or len(query_results) == 0:
            return self._handle_no_results(user_query, sql_query,salesman_response=salesman_response)
        
        return self._handle_results(user_query, sql_query, query_results, user_preferences)

    def _handle_no_results(self, user_query: str, sql_query: str,salesman_response:str) -> str:
        """Handle case when no results are found"""
        last_successful = self.query_history[-1] if self.query_history else None
        try:
            prompt = self._create_no_results_prompt(user_query=user_query, last_successful=last_successful,salesman_response=salesman_response)
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are Salem Abu Mohamed, an experienced real estate agent helping clients find properties. Make natural, helpful suggestions based on available data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception:
            return "I noticed your search didn't find any matches. Let me help you adjust your criteria to find something suitable."

    def _handle_results(self, user_query: str, sql_query: str, 
                       query_results: pd.DataFrame, user_preferences: Dict) -> str:
        """Handle case when results are found"""
        prompt = self._create_results_prompt(user_query, sql_query, query_results, user_preferences)
        print(f"Prompt for results: {prompt}")
        response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are Salem Abu Mohamed, an experienced real estate agent helping clients find properties. Make natural, helpful suggestions based on available data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.45
            )
        return response.choices[0].message.content

    def _create_no_results_prompt(self, user_query: str, last_successful: Dict,salesman_response:str) -> str:
        """Create prompt for handling no results case"""

        sql_query = last_successful['query'] if last_successful else 'No previous query'
        results = last_successful['results'].to_string() if last_successful and 'results' in last_successful else 'No previous results'
        search_constraints = json.dumps(self.search_constraints, indent=2, ensure_ascii=False)
        salesman_response = salesman_response if salesman_response else 'No previous response'

        return self.prompts.NO_RESULT_PROMPT.format(
            user_query=user_query,
            sql_query=sql_query,
            last_successful=results,
            search_constraints=search_constraints,
            salesman_response=salesman_response
        )
        

    def _create_results_prompt(self, user_query: str, sql_query: str,
                           query_results: pd.DataFrame, user_preferences: Dict) -> str:
        """Create prompt for handling results case"""

        # 'user_coordinates'
        # 🔹 Step 1: Define required preferences
        
        if user_preferences.get("user_coordinates"):
            required_prefs = ["listing_type", "unit_types"]
        else:
            required_prefs = ["listing_type", "unit_types", "city", "neighborhood"]
        # 🔹 Step 2: Check which preferences are missing
        missing_prefs = [pref for pref in required_prefs if not user_preferences.get(pref)]
        print(f"Missing preferences: {missing_prefs}")
        print(f"Query results length: {len(query_results)}")

        # 🔹 Step 3: If any preferences are missing, ask clarifying questions
        if missing_prefs and len(query_results) > self.max_results_for_summary:
            questions = []
            #TODO: use actual values from results
            for pref in missing_prefs[:2]:  # Limit to first two missing preferences
                if pref in query_results.columns:
                    unique_vals = query_results[pref].dropna().unique().tolist()
                    if unique_vals:
                        formatted_vals = ", ".join(map(str, unique_vals))
                        questions.append(f"تفضل في {pref}? ({formatted_vals})")

            # Fallback if no unique values are found in results
            if not questions:
                questions = [f"من فضلك حدد {', '.join(missing_prefs)}"]

            return self.prompts.MISING_REQUIRED_PREFERENCES.format(mising_questions='، '.join(questions))


        # 🔹 Step 4: Otherwise, proceed with generating the normal prompt
        return self.prompts.RESULTS_PROMPT.format(
            user_query=user_query,
            no_of_query_results=len(query_results),
            sql_query=sql_query,
            query_results=query_results.to_string(),
            user_preferences= user_preferences,
            search_constraints=json.dumps(self.search_constraints, indent=2, ensure_ascii=False),
            max_results_for_summary=self.max_results_for_summary)
    