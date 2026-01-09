"""GraphQL data fetcher for VacayLanka website.

Fetches data from GraphQL endpoint and formats it as context for RAG.
"""

import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

GRAPHQL_ENDPOINT = "http://vacaylanka.atwebpages.com/graphql"

# GraphQL Queries
QUERIES = {
    "footer": """
        query GetFooterFields {
            ownerInfo {
                office_phone
                email
                office_address
                fb_address
                insta_address
                linkedin_address
                twitter_address
            }
        }
    """,
    "activities": """
        query GetActivities {
            activities(where: {orderby: {field: DATE, order: ASC}}) {
                nodes {
                    title
                    featuredImage {
                        node {
                            sourceUrl
                        }
                    }
                    activityDetails {
                        description
                        icon
                        cardColor
                    }
                }
            }
        }
    """,
    "form_fields": """
        query GetFormFields {
            ownerInfo {
                proprietor_name
                phone_number
                office_phone
                office_address
                whatsapp
                fb_address
                insta_address
                linkedin_address
                twitter_address
                map_embedded_link
            }
        }
    """,
    "destinations": """
        query GetDestinations {
            destinations {
                nodes {
                    title
                    featuredImage {
                        node {
                            sourceUrl
                        }
                    }
                    destinationsDetails {
                        districtOrder
                        shortDescription
                        spot1 {
                            title
                            description
                            image {
                                node {
                                    sourceUrl
                                }
                            }
                        }
                        spot2 {
                            title
                            description
                            image {
                                node {
                                    sourceUrl
                                }
                            }
                        }
                        spot3 {
                            title
                            description
                            image {
                                node {
                                    sourceUrl
                                }
                            }
                        }
                    }
                }
            }
        }
    """,
    "itineraries": """
        query GetItinerariesDesc {
            itineraries(where: {orderby: {field: DATE, order: ASC}}) {
                nodes {
                    title
                    date
                    itineraryDetails {
                        duration
                        tag
                        price
                        description
                        daysList
                        includesList
                        notIncludesList
                    }
                }
            }
        }
    """,
    "packages": """
        query GetPackagesAsc {
            packages(where: {orderby: {field: DATE, order: ASC}}) {
                nodes {
                    title
                    date
                    packageDetails {
                        desc
                        packageIcon
                        packageColor
                        price
                        image {
                            node {
                                sourceUrl
                            }
                        }
                    }
                }
            }
        }
    """
}


class VacayLankaFetcher:
    """Fetches and caches data from VacayLanka GraphQL API."""
    
    def __init__(self, cache_ttl_minutes: int = 30):
        """Initialize the fetcher.
        
        Args:
            cache_ttl_minutes: How long to cache data before refreshing.
        """
        self._cache: Dict[str, Any] = {}
        self._cache_times: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._all_data_cache: Optional[str] = None
        self._all_data_cache_time: Optional[datetime] = None
    
    def _execute_query(self, query_name: str, query: str) -> Optional[Dict[str, Any]]:
        """Execute a single GraphQL query.
        
        Args:
            query_name: Name of the query for logging.
            query: The GraphQL query string.
            
        Returns:
            Query result data or None on error.
        """
        try:
            response = requests.post(
                GRAPHQL_ENDPOINT,
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            response.raise_for_status()
            
            data = response.json()
            if "errors" in data:
                logger.warning(f"GraphQL errors for {query_name}: {data['errors']}")
                return None
            
            return data.get("data", {})
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {query_name}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON response for {query_name}: {e}")
            return None
    
    def fetch_query(self, query_name: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetch data for a specific query with caching.
        
        Args:
            query_name: Name of the query (footer, activities, etc.)
            force_refresh: If True, bypass cache.
            
        Returns:
            Query result data or None.
        """
        if query_name not in QUERIES:
            logger.warning(f"Unknown query: {query_name}")
            return None
        
        # Check cache
        if not force_refresh and query_name in self._cache:
            cache_time = self._cache_times.get(query_name)
            if cache_time and datetime.now() - cache_time < self._cache_ttl:
                logger.debug(f"Using cached data for {query_name}")
                return self._cache[query_name]
        
        # Fetch fresh data
        data = self._execute_query(query_name, QUERIES[query_name])
        
        if data:
            self._cache[query_name] = data
            self._cache_times[query_name] = datetime.now()
        
        return data
    
    def fetch_all(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch all queries and return combined data.
        
        Args:
            force_refresh: If True, bypass all caches.
            
        Returns:
            Dict with all query results.
        """
        results = {}
        for query_name in QUERIES:
            data = self.fetch_query(query_name, force_refresh)
            if data:
                results[query_name] = data
        
        return results
    
    def get_context_for_llm(self, force_refresh: bool = False) -> str:
        """Get all data formatted as context string for LLM.
        
        Args:
            force_refresh: If True, bypass cache and re-fetch.
            
        Returns:
            Formatted context string.
        """
        # Check context cache
        if not force_refresh and self._all_data_cache:
            if self._all_data_cache_time and datetime.now() - self._all_data_cache_time < self._cache_ttl:
                return self._all_data_cache
        
        all_data = self.fetch_all(force_refresh)
        
        if not all_data:
            return "No data available from VacayLanka website."
        
        context_parts = ["=== VacayLanka Travel Information ===\n"]
        
        # Format owner/contact info
        if "footer" in all_data or "form_fields" in all_data:
            owner_info = all_data.get("footer", {}).get("ownerInfo") or \
                        all_data.get("form_fields", {}).get("ownerInfo")
            if owner_info:
                context_parts.append("## Contact Information")
                if owner_info.get("proprietor_name"):
                    context_parts.append(f"Proprietor: {owner_info['proprietor_name']}")
                if owner_info.get("office_phone"):
                    context_parts.append(f"Office Phone: {owner_info['office_phone']}")
                if owner_info.get("phone_number"):
                    context_parts.append(f"Phone: {owner_info['phone_number']}")
                if owner_info.get("whatsapp"):
                    context_parts.append(f"WhatsApp: {owner_info['whatsapp']}")
                if owner_info.get("email"):
                    context_parts.append(f"Email: {owner_info['email']}")
                if owner_info.get("office_address"):
                    context_parts.append(f"Address: {owner_info['office_address']}")
                context_parts.append("")
        
        # Format activities
        if "activities" in all_data:
            activities = all_data["activities"].get("activities", {}).get("nodes", [])
            if activities:
                context_parts.append("## Activities Available")
                for act in activities:
                    title = act.get("title", "Untitled")
                    details = act.get("activityDetails", {})
                    desc = details.get("description", "")
                    context_parts.append(f"- **{title}**: {desc}")
                context_parts.append("")
        
        # Format destinations
        if "destinations" in all_data:
            destinations = all_data["destinations"].get("destinations", {}).get("nodes", [])
            if destinations:
                context_parts.append("## Destinations")
                for dest in destinations:
                    title = dest.get("title", "Untitled")
                    details = dest.get("destinationsDetails", {})
                    short_desc = details.get("shortDescription", "")
                    context_parts.append(f"\n### {title}")
                    if short_desc:
                        context_parts.append(short_desc)
                    
                    # Add spots
                    for spot_key in ["spot1", "spot2", "spot3"]:
                        spot = details.get(spot_key, {})
                        if spot and spot.get("title"):
                            context_parts.append(f"  - {spot['title']}: {spot.get('description', '')}")
                context_parts.append("")
        
        # Format packages
        if "packages" in all_data:
            packages = all_data["packages"].get("packages", {}).get("nodes", [])
            if packages:
                context_parts.append("## Travel Packages")
                for pkg in packages:
                    title = pkg.get("title", "Untitled")
                    details = pkg.get("packageDetails", {})
                    desc = details.get("desc", "")
                    price = details.get("price", "")
                    context_parts.append(f"- **{title}**")
                    if price:
                        context_parts.append(f"  Price: {price}")
                    if desc:
                        context_parts.append(f"  {desc}")
                context_parts.append("")
        
        # Format itineraries
        if "itineraries" in all_data:
            itineraries = all_data["itineraries"].get("itineraries", {}).get("nodes", [])
            if itineraries:
                context_parts.append("## Itineraries")
                for itin in itineraries:
                    title = itin.get("title", "Untitled")
                    details = itin.get("itineraryDetails", {})
                    duration = details.get("duration", "")
                    price = details.get("price", "")
                    desc = details.get("description", "")
                    tag = details.get("tag", "")
                    
                    context_parts.append(f"\n### {title}")
                    if tag:
                        context_parts.append(f"Tag: {tag}")
                    if duration:
                        context_parts.append(f"Duration: {duration}")
                    if price:
                        context_parts.append(f"Price: {price}")
                    if desc:
                        context_parts.append(desc)
                    
                    days_list = details.get("daysList", "")
                    if days_list:
                        context_parts.append(f"Days: {days_list}")
                    
                    includes = details.get("includesList", "")
                    if includes:
                        context_parts.append(f"Includes: {includes}")
                    
                    not_includes = details.get("notIncludesList", "")
                    if not_includes:
                        context_parts.append(f"Not Included: {not_includes}")
                context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Cache the formatted context
        self._all_data_cache = context
        self._all_data_cache_time = datetime.now()
        
        logger.info(f"Fetched VacayLanka context ({len(context)} chars)")
        return context
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._cache_times.clear()
        self._all_data_cache = None
        self._all_data_cache_time = None
        logger.info("VacayLanka cache cleared")


# Global fetcher instance
_vacaylanka_fetcher: Optional[VacayLankaFetcher] = None


def get_vacaylanka_fetcher() -> VacayLankaFetcher:
    """Get or create the global VacayLanka fetcher."""
    global _vacaylanka_fetcher
    if _vacaylanka_fetcher is None:
        _vacaylanka_fetcher = VacayLankaFetcher()
    return _vacaylanka_fetcher
