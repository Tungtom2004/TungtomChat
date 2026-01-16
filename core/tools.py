# core/tools.py
import os
import datetime
import pytz

from serpapi import GoogleSearch
from langchain_core.tools import tool

# ===============================
# TOOL: CURRENT TIME
# ===============================
@tool
def get_current_time(timezone: str = "Asia/Ho_Chi_Minh") -> str:
    """
    Return the current time in Vietnamese.
    """
    try:
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)

        weekdays = {
            "Monday": "Thứ Hai",
            "Tuesday": "Thứ Ba",
            "Wednesday": "Thứ Tư",
            "Thursday": "Thứ Năm",
            "Friday": "Thứ Sáu",
            "Saturday": "Thứ Bảy",
            "Sunday": "Chủ Nhật",
        }

        wd_vi = weekdays.get(now.strftime("%A"), "Thứ")
        date_vi = now.strftime(f"{wd_vi}, ngày %d tháng %m năm %Y")
        time_vi = now.strftime("%H:%M:%S")

        return f"Thời gian hiện tại ở {timezone} là {time_vi} vào {date_vi}."
    except Exception as e:
        return f"Lỗi khi lấy thời gian hiện tại: {e}"

# ===============================
# TOOL: GOOGLE SEARCH
# ===============================
@tool
def google_search(query: str) -> str:
    """
    Used when the user asks for general information, facts, definitions,
    or anything that is not part of the main PTIT subjects.
    """
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "Thiếu SERPAPI_API_KEY."

        params = {
            "q": query,
            "api_key": api_key,
            "location": "Vietnam",
            "gl": "vn",
            "hl": "vi",
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        snippets = []
        if "organic_results" in results:
            for res in results["organic_results"][:3]:
                if "snippet" in res:
                    snippets.append(res["snippet"])

        if not snippets:
            return f"Không tìm thấy kết quả cho truy vấn: {query}"

        return "Dưới đây là các kết quả tóm tắt tìm được:\n- " + "\n- ".join(snippets)

    except Exception as e:
        return f"Lỗi khi thực hiện tìm kiếm Google: {e}"

# ===============================
# TOOL REGISTRY
# ===============================
GENERAL_TOOLS = [
    get_current_time,
    google_search,
]
