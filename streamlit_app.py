import streamlit as st
from services.real_estate_manager import RealEstateManager
from typing import Dict, Any, List
import json

class RealEstateStreamlitApp:
    def __init__(self):
        self.manager = RealEstateManager()
        self.init_session_state()
        self.MAX_RESULTS_FOR_SUMMARY = 2  # Matches the value in AIManager

    def display_image_carousel(self, image_urls: List[Any], container):
        """Display a carousel of images with navigation"""
        if not image_urls:
            return
        print(f"Displaying {image_urls} in carousel")
        # Flatten all galleries into a single list of URLs
        flat_urls = []
        for item in image_urls:
            if isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, list):
                        flat_urls.extend(parsed)
                    else:
                        flat_urls.append(parsed)
                except json.JSONDecodeError:
                    flat_urls.append(item)
            elif isinstance(item, list):
                flat_urls.extend(item)

        if not flat_urls:
            return

        # Initialize carousel state if not exists
        if 'carousel_index' not in st.session_state:
            st.session_state.carousel_index = 0

        # Keep index in range
        st.session_state.carousel_index = max(0, min(st.session_state.carousel_index, max(0, len(flat_urls) - 3)))

        total_images = len(flat_urls)
        
        # Create columns for navigation and images
        left_arrow, img1, img2, img3, right_arrow = container.columns([1, 3, 3, 3, 1])

        # Left arrow
        with left_arrow:
            st.markdown("<div style='height: 150px; display: flex; align-items: center; justify-content: center;'>", unsafe_allow_html=True)
            if st.button("◀", key="left_arrow") and st.session_state.carousel_index > 0:
                st.session_state.carousel_index -= 1
            st.markdown("</div>", unsafe_allow_html=True)

        # Display 3 images
        for i, col in enumerate([img1, img2, img3]):
            with col:
                idx = st.session_state.carousel_index + i
                if idx < total_images:
                    st.markdown(
                        f"<div style='height: 150px; display: flex; align-items: center; justify-content: center;'>"
                        f"<img src='{flat_urls[idx]}' style='max-height: 150px; width: auto;'/>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        # Right arrow
        with right_arrow:
            st.markdown("<div style='height: 150px; display: flex; align-items: center; justify-content: center;'>", unsafe_allow_html=True)
            if st.button("▶", key="right_arrow") and st.session_state.carousel_index < max(0, total_images - 3):
                st.session_state.carousel_index += 1
            st.markdown("</div>", unsafe_allow_html=True)

        # Display image counter
        container.markdown(
            f"<div style='text-align: center; margin-top: 10px;'>"
            f"Showing images {st.session_state.carousel_index + 1}-{min(st.session_state.carousel_index + 3, total_images)} "
            f"of {total_images}</div>",
            unsafe_allow_html=True
        )

    def init_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            # Initialize user preferences with the structure from RealEstateManager
            st.session_state.user_preferences = self.manager.user_preferences
            # Initialize carousel state
            st.session_state.carousel_index = 0
            
            # Initialize chat memory and messages
            st.session_state.chat_memory = []
            st.session_state.messages = []
            
            # Store search constraints in session state
            st.session_state.search_constraints = self.manager.db_manager.get_search_constraints()
            
            st.session_state.initialized = True

    def create_sidebar(self):
        """Create and manage sidebar elements"""
        with st.sidebar:
            st.header("Debug Panel")
            
            # Initialize toggle states
            if 'show_preferences' not in st.session_state:
                st.session_state.show_preferences = False
            if 'show_constraints' not in st.session_state:
                st.session_state.show_constraints = False
            
            # Toggle buttons
            st.session_state.show_preferences = st.checkbox(
                "Show User Preferences", 
                value=st.session_state.show_preferences
            )
            st.session_state.show_constraints = st.checkbox(
                "Show Search Constraints", 
                value=st.session_state.show_constraints
            )
            
            # Display preferences and constraints
            if st.session_state.show_preferences:
                st.subheader("User Preferences")
                st.json(st.session_state.user_preferences)
            
            if st.session_state.show_constraints:
                st.subheader("Search Constraints")
                st.json(st.session_state.search_constraints)
                
                # Add a refresh button for constraints
                if st.button("Refresh Constraints"):
                    st.session_state.search_constraints = self.manager.db_manager.get_search_constraints()
                    st.rerun()
            
            # Database connection status
            if self.manager.db_manager.db_conn:
                st.success("✅ Database Connected")
            else:
                st.error("❌ Database Not Connected")
            
            # Display database schema
            if st.button("Show Database Schema"):
                st.text_area("Database Schema", self.manager.db_manager.get_database_schema(), height=200)
            
            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.session_state.chat_memory = []
                self.manager.reset_preferences()
                st.rerun()

    def process_message(self, prompt: str):
        """Process a single user message"""
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt.strip())
        
        # Process with manager
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get response from manager 
                response = self.manager.process_user_message(prompt,user_preferences=st.session_state.user_preferences,conversation_memory=st.session_state.chat_memory)
                
                # Update session state with current preferences
                st.session_state.user_preferences = self.manager.get_current_preferences()
                
                # Display response
                st.markdown(response)
                
                # If there are recent query results in history, display them
                if self.manager.ai_manager.query_history:
                    latest_query = self.manager.ai_manager.query_history[-1]
                    query_results = latest_query.get('results')
                    
                    if query_results is not None and not query_results.empty:
                        # Only show detailed results if within summary limit
                        if len(query_results) <= self.MAX_RESULTS_FOR_SUMMARY:
                            # Drop unnecessary columns
                            display_results = query_results.drop(
                                columns=['project_id', 'publisher_id', 'coordinates', 
                                        'latitude', 'longitude', 'created_at'], 
                                errors='ignore'
                            )
                            display_results = display_results.drop(
                                                    columns=[
                                                        col for col in display_results.columns
                                                        if ((display_results[col].isna()) | (display_results[col] == 0)).all()
                                                    ]
                                                )
                            st.dataframe(display_results)
                            ## send the list of images in column gallery in display_results to  display_image_carousel
                            if 'gallery' in display_results.columns:
                                image_urls = []
                                for g in display_results['gallery'].dropna():
                                    if isinstance(g, str):
                                        try:
                                            parsed = json.loads(g)
                                            if isinstance(parsed, list):
                                                image_urls.extend(parsed)  # Add all URLs in the list
                                            else:
                                                image_urls.append(parsed)  # Single URL
                                        except json.JSONDecodeError:
                                            image_urls.append(g)  # If not JSON, treat as plain URL
                                    elif isinstance(g, list):
                                        image_urls.extend(g)

                                if image_urls:
                                    cols = st.columns(3)
                                    for col, img_url in zip(cols, image_urls[:3]):
                                        col.image(img_url, use_column_width=True)
                            
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "data": query_results if 'query_results' in locals() else None
                })

    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(page_title="Real Estate Chatbot", layout="wide")
        st.title("🏘️ مشاريع راكز العقارية")
        st.markdown("<h3 style='text-align: center;'>مستقبلك في عقار موثوق</h3>", unsafe_allow_html=True)
        
        # Display Salem's image and story
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("salem_image.webp", caption="Salem Abu Mohamed")
        
        with col2:
            st.markdown("""
            <div style='font-size: 1.2em; direction: rtl;'>
            <h2>🧔‍♂️ قصة سالم أبو محمد – أسطورة العقار</h2>
            سالم أبو محمد، رجل بدأ رحلته في العقار من على "دكة" في أحد أحياء الرياض قبل 25 سنة، وكان وقتها يبيع الأراضي على ورق كروكي مرسوم باليد! الناس كانت تقول له "يا سالم، السوق نايم"، لكنه كان يرد بثقة: "السوق ما ينام، اللي ينام اللي ما يفهم السوق!"

            من بيع أول شقة غرفتين وصالة، إلى صفقات بملايين الريالات، سالم ما فوت ولا فرصة. يُقال إنه باع فيلا في شمال الرياض لرجل ما كان ناوي يشتري، بس بعد ما جلس مع سالم نص ساعة طلع كاش وكتب الشيك! عنده قدرة خارقة يقنع أي أحد إن العقار هو المستقبل، ويمكن الماضي كمان.

            سالم يعرف كل شارع، كل زاوية، وكل طوبة في المدينة. حتى GPS يستشيره أحيانًا! الزبائن يثقون فيه لأنهم يعرفون: "إذا سالم قالك هذا العقار فرصتك… صدقه، ولو السعر غالي!"

            والأطرف؟ سالم مرة باع بيت لشخص قبل لا يخلص بناءه، والزبون طلع يحلف إن الحلم اللي شافه هو نفس أوصاف سالم!

            الناس ما تقول "الخبير العقاري سالم" عبث… لأنه ببساطة: العقار يعرفه قبل لا هو يعرف العقار.
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Create sidebar
        self.create_sidebar()
        
        # Main chat interface
        st.header("Chat with Salem Abu Mohamed")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "data" in message and message["data"] is not None and len(message["data"]) <= self.MAX_RESULTS_FOR_SUMMARY:
                    st.dataframe(message["data"])
        
        # Chat input
        if prompt := st.chat_input("اسأل عن المشاريع العقارية..."):
            self.process_message(prompt)

def main():
    app = RealEstateStreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
