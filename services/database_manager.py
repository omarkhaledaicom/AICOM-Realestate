import psycopg2
import pandas as pd
import math
from typing import Dict, List, Optional
from decimal import Decimal
import re
class DatabaseManager:


    def __init__(self, db_config: Dict):
        self.db_config = db_config 
        self.db_conn = self.init_database_connection()

    def _convert_decimal_to_str(self, value):
        """Convert Decimal objects to strings for JSON serialization"""
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, list):
            return [self._convert_decimal_to_str(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._convert_decimal_to_str(v) for k, v in value.items()}
        return value
    def _extract_numbers_from_area(self,area_str):
        """Extract numeric values from area strings like '80 - 120 م²'."""
        if pd.isna(area_str):
            return None
        nums = re.findall(r'\d+', str(area_str))
        return [int(n) for n in nums] if nums else None
    
    def _create_ranges(self,values, num_ranges=5):
        """Create numeric ranges from a sorted list of numbers."""
        if not values:
            return []
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return [f"{int(min_val)}"]  # All same value
        step = (max_val - min_val) / num_ranges
        ranges = []
        start = min_val
        for i in range(num_ranges):
            end = min_val + step * (i + 1)
            ranges.append(f"{int(start)} - {int(end)}")
            start = end + 1
        return ranges
    def _describe_rooms(self,values):
        """Describe discrete numeric values for rooms, showing missing numbers."""
        if not values:
            return []
        values = sorted(set(int(v) for v in values))
        min_val, max_val = values[0], values[-1]
        missing = [x for x in range(min_val, max_val + 1) if x not in values]
        description = f"{min_val} - {max_val}"
        if missing:
            missing_str = ", ".join(map(str, missing))
            description += f" (missing {missing_str})"
        return [description]
    # Database credentials

    def init_database_connection(self):
        """Initialize database connection and return connection object"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")

    def get_search_constraints(self, city=None) -> Dict[str, List]:
        """Get unique values for each column from database"""
        if not self.db_conn:
            # Mock data for testing when DB is not available
            return {
                'project_status': ['متاح', 'مكتمل', 'تحت الإنشاء'],
                'unit_types': ['شقق', 'فلل', 'تاون هاوس'],
                'city': ['الرياض', 'جدة', 'الدمام'],
                'price_from': ['139,000', '200,000', '300,000', '500,000'],
                'rooms': [1, 2, 3, 4, 5],
                'area': ['80 - 120 م²', '121 - 154 م²', '155 - 200 م²']
            }

        constraints = {}
        cursor = None
        try:
            cursor = self.db_conn.cursor()
            columns = ['listing_type','project_status', 'unit_types', 'city', 'neighborhood', 'price', 'rooms', 'area']
            MAX_VALUES = 10

            for column in columns:
                try:
                    sql_query = f"SELECT DISTINCT {column} FROM rakez_projects_cleaned WHERE {column} IS NOT NULL"
                    if city:
                        sql_query += f" AND city='{city}'"
                    cursor.execute(sql_query)
                    values = [row[0] for row in cursor.fetchall()]
                    values = [v for v in values if v is not None]

                    if column == "price":
                        numeric_vals = sorted(set(float(v) for v in values if str(v).replace('.', '', 1).isdigit()))
                        constraints[column] = self._create_ranges(numeric_vals, num_ranges=5)

                    elif column == "rooms":
                        numeric_vals = sorted(set(int(v) for v in values if str(v).isdigit()))
                        constraints[column] = self._describe_rooms(numeric_vals)

                    elif column == "area":
                        numeric_vals = []
                        for v in values:
                            nums = self._extract_numbers_from_area(v)
                            if nums:
                                numeric_vals.extend(nums)
                        numeric_vals = sorted(set(numeric_vals))
                        constraints[column] = self._create_ranges(numeric_vals, num_ranges=5)

                    else:
                        values = [self._convert_decimal_to_str(v) for v in values]
                        if len(values) > MAX_VALUES:
                            values = values[:MAX_VALUES] + [f"و {len(values) - MAX_VALUES} خيارات اخرى"]
                        constraints[column] = values

                except Exception as e:
                    print(f"Could not fetch values for {column}: {e}")
                    constraints[column] = []
                    self.db_conn.rollback()

            self.db_conn.commit()
            return constraints

        except Exception as e:
            if self.db_conn:
                self.db_conn.rollback()
            raise Exception(f"Error getting search constraints: {e}")
        finally:
            if cursor:
                cursor.close()

    def execute_sql_query(self, query: str) -> Optional[pd.DataFrame]:
        """Execute SQL query and return results as DataFrame"""
        if not self.db_conn or not query:
            return None
        
        try:
            df = pd.read_sql_query(query, self.db_conn)
            # Convert any Decimal columns to float or string
            for column in df.columns:
                if df[column].dtype.name == 'object':
                    # Check if the column contains any Decimal objects
                    if any(isinstance(x, Decimal) for x in df[column].dropna()):
                        df[column] = df[column].apply(lambda x: str(x) if isinstance(x, Decimal) else x)
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    def get_database_schema(self) -> str:
        """Retrieve and format database schema information"""
        if not self.db_conn:
            return """
            Table name: rakez_projects_cleaned
            Columns: project_id (bigint), publisher_id (double precision), project_status (text), 
                    unit_types (text), city (text), price_from (text), rooms (bigint), 
                    area (text), near_by (text), gallery (text), project_url (text)
            """
        
        try:
            cursor = self.db_conn.cursor()
            schema_query = """
            SELECT 
                table_name,
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name IN ('rakez_projects_cleaned')
            ORDER BY table_name, ordinal_position;
            """
            
            cursor.execute(schema_query)
            schema_info = cursor.fetchall()
            
            if not schema_info:
                return "No schema information found."
            
            schema_text = "Database Schema:\n\n"
            current_table = None
            
            for row in schema_info:
                table_name, column_name, data_type, is_nullable, column_default = row
                
                if current_table != table_name:
                    if current_table is not None:
                        schema_text += "\n"
                    schema_text += f"Table name: {table_name}\nColumns:\n"
                    current_table = table_name
                
                nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
                default = f", DEFAULT: {column_default}" if column_default else ""
                schema_text += f"  - {column_name} ({data_type}, {nullable}{default})\n"
            
            return schema_text
            
        except Exception as e:
            raise Exception(f"Error retrieving database schema: {e}")

    def get_nearby_places_for_projects(self, project_ids):
        """
        Given a list of project_ids, return up to 2 nearby places per type 
        (restaurant, hospital, bank, school) for each project, as dicts.
        """
        if not self.db_conn:
            return []
        cursor = None
        try:
            cursor = self.db_conn.cursor()
            query = """
                WITH ranked_places AS (
                    SELECT 
                        up.project_id,
                        np.place_id,
                        np.name,
                        np.type,
                        np.rating,
                        np.address,
                        np.lat,
                        np.lng,
                        ROW_NUMBER() OVER (
                            PARTITION BY up.project_id, np.type
                            ORDER BY np.rating DESC NULLS LAST, np.name
                        ) AS rn
                    FROM unit_nearby_places up
                    JOIN nearby_places np ON up.place_id = np.place_id
                    WHERE up.project_id = ANY(%s)
                    AND np.type IN ('restaurant', 'hospital', 'bank', 'school','gym','shopping_mall','mosque')
                )
                SELECT 
                    project_id,
                    place_id,
                    name,
                    type,
                    rating,
                    address,
                    lat,
                    lng
                FROM ranked_places
                WHERE rn <= 2
                ORDER BY project_id, type, rating DESC, name;
            """
            cursor.execute(query, (project_ids,))
            colnames = [desc[0] for desc in cursor.description]
            return [dict(zip(colnames, row)) for row in cursor.fetchall()]
        except Exception as e:
            if self.db_conn:
                self.db_conn.rollback()
            raise Exception(f"Error getting nearby places: {e}")
        finally:
            if cursor:
                cursor.close()




    def haversine(self,lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
            math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

    def get_closest_k_units(self, center_lat, center_lng, radius_km=None, top_k=10):
        if not self.db_conn:
            return []
        cursor = None
        try:
            cursor = self.db_conn.cursor()

            # Base SQL with Haversine formula
            base_sql = """
                SELECT 
                    project_id,
                    coordinates,
                    (
                        6371 * acos(
                            cos(radians(%s)) 
                            * cos(radians(split_part(trim(both '()' from coordinates), ',', 1)::float)) 
                            * cos(radians(split_part(trim(both '()' from coordinates), ',', 2)::float) - radians(%s)) 
                            + sin(radians(%s)) 
                            * sin(radians(split_part(trim(both '()' from coordinates), ',', 1)::float))
                        )
                    ) AS distance_km
                FROM rakez_projects_cleaned
                WHERE coordinates IS NOT NULL
                {radius_filter}
                ORDER BY distance_km ASC
                LIMIT %s;
            """

            params = [center_lat, center_lng, center_lat]

            if radius_km is not None:
                # Radius filter using same formula in WHERE
                radius_filter = """AND (
                        6371 * acos(
                            cos(radians(%s)) 
                            * cos(radians(split_part(trim(both '()' from coordinates), ',', 1)::float)) 
                            * cos(radians(split_part(trim(both '()' from coordinates), ',', 2)::float) - radians(%s)) 
                            + sin(radians(%s)) 
                            * sin(radians(split_part(trim(both '()' from coordinates), ',', 1)::float))
                        )
                    ) <= %s"""
                
                params.extend([center_lat, center_lng, center_lat, radius_km])
            else:
                radius_filter = ""

            sql = base_sql.format(radius_filter=radius_filter)

            params.append(top_k)

            print(f"Executing SQL: {sql} with params: {params}")
            cursor.execute(sql, params)

            results = cursor.fetchall()
            return [
                {
                    "project_id": row[0],
                    "coordinates": row[1],
                    "distance_km": round(row[2], 3)
                }
                for row in results
            ]

        except Exception as e:
            if self.db_conn:
                self.db_conn.rollback()
            raise Exception(f"Error getting closest units: {e}")
        finally:
            if cursor:
                cursor.close()

