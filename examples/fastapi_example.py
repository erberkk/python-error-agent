from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
from error_agent import ErrorAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize ErrorAgent
error_agent = ErrorAgent(
    llm_url=os.getenv("LLM_URL", "http://localhost:11434"),
    project_root=os.getcwd(),  # Index THIS project (where fastapi_example.py is)
    model=os.getenv("LLM_MODEL", "llama3:8b"),
    # Slack configuration (optional)
    slack_token=os.getenv("SLACK_TOKEN"),
    slack_channel=os.getenv("SLACK_CHANNEL"),
    # Google Chat configuration (optional)
    google_chat_webhook=os.getenv("GOOGLE_CHAT_WEBHOOK"),
    app_name=os.getenv("APP_NAME", "Error Agent"),
    # Security and privacy
    require_local_llm=os.getenv("REQUIRE_LOCAL_LLM", "true").lower() == "true",
    # Index configuration - focus on user's project files
    index_include=[
        p.strip()
        for p in os.getenv("INDEX_INCLUDE", "*.py,**/*.py").split(",")
        if p.strip()
    ],
    index_exclude=[
        p.strip()
        for p in os.getenv(
            "INDEX_EXCLUDE",
            "**/error_agent/**,**/tests/**,**/venv/**,**/.venv/**,**/__pycache__/**,**/node_modules/**",
        ).split(",")
        if p.strip()
    ],
    index_lazy=True,
    index_background=True,
    auto_apply_fixes=os.getenv("AUTO_APPLY_FIXES", "true").lower() == "true",
    auto_lint_after_apply=os.getenv("AUTO_LINT_AFTER_APPLY", "true").lower() == "true",
    auto_open_github_pr=os.getenv("AUTO_OPEN_GITHUB_PR", "true").lower() == "true",
    github_token=os.getenv("GITHUB_TOKEN"),
    branch_name_for_auto_github_pr=os.getenv("BRANCH_NAME_FOR_AUTO_GITHUB_PR"),
)

# Install error handler
error_agent.install()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions."""
    logger.info(f"Global exception handler caught: {type(exc).__name__}: {str(exc)}")

    # Submit to background worker; do not block the response cycle
    error_agent.submit_exception(type(exc), exc, exc.__traceback__)

    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "message": str(exc),
            "path": request.url.path,
        },
    )


@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Welcome to the FastAPI example!"}


@app.get("/error")
async def trigger_error():
    """
    Endpoint that triggers a ZeroDivisionError for testing.

    This endpoint is used to test the error handling functionality.
    """
    result = 1 / 0
    return {"result": result}


@app.get("/simple-error-test")
async def simple_error_test():
    data = {"existing_key": "value"}

    missing_value = (
        "default_value" if "missing_key" not in data else data["missing_key"]
    )

    return {"result": missing_value}


@app.get("/nested-error")
async def nested_error():
    """
    This endpoint demonstrates a complex error scenario with nested function calls
    and multiple layers of error propagation.
    """

    def process_user_data(user_id: str, data: dict) -> dict:
        """Process user data and return formatted result."""
        try:
            preferences = data["preferences"]
            return format_user_preferences(user_id, preferences)
        except KeyError as e:
            raise ValueError(f"Invalid user data format: missing {str(e)}") from e

    def format_user_preferences(user_id: str, preferences: dict) -> dict:
        """
        Format user preferences with additional metadata.

        Args:
            user_id: The ID of the user
            preferences: Dictionary containing user preferences

        Returns:
            Dictionary containing formatted user preferences with settings

        Raises:
            ValueError: If preferences format is invalid or theme is not a string
        """
        try:
            if not isinstance(user_id, str):
                raise ValueError("User ID must be a string")
            if not isinstance(preferences, dict):
                raise ValueError("Preferences must be a dictionary")

            theme = preferences.get("theme")
            if theme is None:
                raise ValueError("Theme is required in preferences")
            if not isinstance(theme, str):
                raise ValueError(f"Theme must be a string, got {type(theme).__name__}")

            settings = get_user_settings(theme)

            return {"user_id": user_id, "theme": theme, "settings": settings}
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid preferences format: {str(e)}") from e

    def get_user_settings(theme: str) -> dict:
        """Get user settings based on theme."""
        if not isinstance(theme, str):
            raise ValueError("Theme must be a string")

        settings = {
            "light": {"background": "white", "text": "black"},
            "dark": {"background": "black", "text": "white"},
        }

        if theme not in settings:
            raise ValueError(
                f"Invalid theme: {theme}. Available themes: {list(settings.keys())}"
            )

        return settings[theme]

    user_data = {"preferences": {"theme": 123}}

    return process_user_data("user123", user_data)


@app.get("/large-function-error")
async def large_function_error():
    """
    This endpoint demonstrates error handling in a large function (300+ lines).
    The error occurs in a specific block, and the LLM should provide a targeted fix.
    """

    def massive_data_processor(data_input: dict) -> dict:
        """
        A large function that processes complex data through multiple stages.
        This simulates a real-world scenario with a large function where
        an error occurs in a specific section.
        """
        print("Starting data processing...")
        result = {"processed": True, "stages": []}

        if not isinstance(data_input, dict):
            raise TypeError("Input must be a dictionary")

        processed_count = 0
        error_count = 0
        warning_count = 0

        user_data = data_input.get("user", {})
        if not user_data:
            warning_count += 1
            result["stages"].append(
                {
                    "stage": "user_validation",
                    "status": "warning",
                    "message": "No user data",
                }
            )
        else:
            user_id = user_data.get("id")
            if not user_id:
                error_count += 1
                result["stages"].append(
                    {
                        "stage": "user_validation",
                        "status": "error",
                        "message": "Missing user ID",
                    }
                )
            else:
                result["stages"].append(
                    {"stage": "user_validation", "status": "success"}
                )
                processed_count += 1

        products = data_input.get("products", [])
        if not isinstance(products, list):
            error_count += 1
            result["stages"].append(
                {
                    "stage": "product_processing",
                    "status": "error",
                    "message": "Products must be a list",
                }
            )
        else:
            processed_products = []
            for idx, product in enumerate(products):
                try:
                    if not isinstance(product, dict):
                        error_count += 1
                        continue

                    required_fields = ["id", "name", "price", "category"]
                    missing_fields = [
                        field for field in required_fields if field not in product
                    ]
                    if missing_fields:
                        error_count += 1
                        result["stages"].append(
                            {
                                "stage": "product_processing",
                                "status": "error",
                                "message": f"Product {idx} missing fields: {missing_fields}",
                            }
                        )
                        continue

                    price = product["price"]
                    if not isinstance(price, (int, float)):
                        try:
                            price = float(price)
                        except (ValueError, TypeError):
                            error_count += 1
                            continue

                    valid_categories = [
                        "electronics",
                        "clothing",
                        "books",
                        "home",
                        "sports",
                    ]
                    category = product["category"].lower()
                    if category not in valid_categories:
                        warning_count += 1
                        category = "other"

                    discount_rate = 0.1 if category == "electronics" else 0.05
                    discounted_price = price * (1 - discount_rate)

                    processed_product = {
                        "id": product["id"],
                        "name": product["name"],
                        "original_price": price,
                        "discounted_price": discounted_price,
                        "category": category,
                        "discount_rate": discount_rate,
                    }
                    processed_products.append(processed_product)
                    processed_count += 1

                except Exception as e:
                    error_count += 1
                    result["stages"].append(
                        {
                            "stage": "product_processing",
                            "status": "error",
                            "message": f"Error processing product {idx}: {str(e)}",
                        }
                    )

            result["processed_products"] = processed_products
            result["stages"].append(
                {
                    "stage": "product_processing",
                    "status": "success",
                    "count": len(processed_products),
                }
            )

        orders = data_input.get("orders", [])
        if orders:
            processed_orders = []
            for order_idx, order in enumerate(orders):
                try:
                    if not isinstance(order, dict):
                        error_count += 1
                        continue

                    order_id = order.get("order_id")
                    if not order_id:
                        error_count += 1
                        continue

                    items = order.get("items", [])
                    if not isinstance(items, list):
                        error_count += 1
                        continue

                    total_amount = 0
                    processed_items = []

                    for item_idx, item in enumerate(items):
                        if not isinstance(item, dict):
                            error_count += 1
                            continue

                        item_id = item.get("product_id")
                        quantity = item.get("quantity", 1)

                        if not item_id:
                            error_count += 1
                            continue

                        product_found = None
                        for product in processed_products:
                            if product["id"] == item_id:
                                product_found = product
                                break

                        if not product_found:
                            error_count += 1
                            result["stages"].append(
                                {
                                    "stage": "order_processing",
                                    "status": "error",
                                    "message": f"Product {item_id} not found in processed products",
                                }
                            )
                            continue

                        item_price = product_found["discounted_price"]
                        item_total = item_price * quantity
                        total_amount += item_total

                        processed_item = {
                            "product_id": item_id,
                            "product_name": product_found["name"],
                            "quantity": quantity,
                            "unit_price": item_price,
                            "total_price": item_total,
                        }
                        processed_items.append(processed_item)

                    tax_rate = 0.08
                    tax_amount = total_amount * tax_rate
                    shipping_fee = 10.0 if total_amount < 50 else 0.0
                    final_amount = total_amount + tax_amount + shipping_fee

                    processed_order = {
                        "order_id": order_id,
                        "items": processed_items,
                        "subtotal": total_amount,
                        "tax_amount": tax_amount,
                        "shipping_fee": shipping_fee,
                        "total_amount": final_amount,
                    }
                    processed_orders.append(processed_order)
                    processed_count += 1

                except Exception as e:
                    error_count += 1
                    result["stages"].append(
                        {
                            "stage": "order_processing",
                            "status": "error",
                            "message": f"Error processing order {order_idx}: {str(e)}",
                        }
                    )

            result["processed_orders"] = processed_orders
            result["stages"].append(
                {
                    "stage": "order_processing",
                    "status": "success",
                    "count": len(processed_orders),
                }
            )

        analytics = {}
        try:
            total_products = (
                len(processed_products) if "processed_products" in result else 0
            )
            total_orders = len(processed_orders) if "processed_orders" in result else 0

            total_revenue = 0
            if "processed_orders" in result:
                for order in result["processed_orders"]:
                    total_revenue += order["total_amount"]

            category_stats = {}
            if "processed_products" in result:
                for product in result["processed_products"]:
                    category = product["category"]
                    if category not in category_stats:
                        category_stats[category] = {
                            "count": 0,
                            "total_value": 0,
                            "avg_price": 0,
                        }
                    category_stats[category]["count"] += 1
                    category_stats[category]["total_value"] += product[
                        "discounted_price"
                    ]

                for category in category_stats:
                    count = category_stats[category]["count"]
                    if count > 0:
                        category_stats[category]["avg_price"] = (
                            category_stats[category]["total_value"] / count
                        )

            customer_analytics = {}
            if "processed_orders" in result:
                customer_orders = {}
                for order in result["processed_orders"]:
                    customer_id = f"customer_{hash(order['order_id']) % 100}"
                    if customer_id not in customer_orders:
                        customer_orders[customer_id] = {
                            "order_count": 0,
                            "total_spent": 0,
                            "orders": [],
                        }
                    customer_orders[customer_id]["order_count"] += 1
                    customer_orders[customer_id]["total_spent"] += order["total_amount"]
                    customer_orders[customer_id]["orders"].append(order["order_id"])

                customer_analytics = customer_orders

            analytics = {
                "summary": {
                    "total_products": total_products,
                    "total_orders": total_orders,
                    "total_revenue": total_revenue,
                    "processed_items": processed_count,
                    "error_count": error_count,
                    "warning_count": warning_count,
                },
                "category_breakdown": category_stats,
                "customer_analytics": customer_analytics,
            }

            result["analytics"] = analytics
            result["stages"].append({"stage": "analytics", "status": "success"})

        except Exception as e:
            error_count += 1
            result["stages"].append(
                {
                    "stage": "analytics",
                    "status": "error",
                    "message": f"Analytics calculation failed: {str(e)}",
                }
            )

        required_result_fields = ["processed", "stages"]
        for field in required_result_fields:
            if field not in result:
                raise ValueError(f"Missing required result field: {field}")

        data_quality_score = 100
        if error_count > 0:
            data_quality_score -= error_count * 10
        if warning_count > 0:
            data_quality_score -= warning_count * 5

        data_quality_score = max(0, data_quality_score)

        problematic_data = data_input.get("quality_config", {})

        quality_threshold = problematic_data["missing_key"]

        if data_quality_score < quality_threshold:
            result["quality_warning"] = (
                f"Data quality score {data_quality_score} below threshold {quality_threshold}"
            )

        result["data_quality_score"] = data_quality_score
        result["final_status"] = "completed"

        result["performance_metrics"] = {
            "processed_items": processed_count,
            "error_rate": error_count / max(1, processed_count) * 100,
            "warning_rate": warning_count / max(1, processed_count) * 100,
        }

        if error_count == 0:
            result["overall_status"] = "success"
        elif error_count < 5:
            result["overall_status"] = "partial_success"
        else:
            result["overall_status"] = "failed"

        result["stages"].append({"stage": "final_validation", "status": "success"})

        result["summary"] = {
            "total_stages": len(result["stages"]),
            "successful_stages": len(
                [s for s in result["stages"] if s["status"] == "success"]
            ),
            "failed_stages": len(
                [s for s in result["stages"] if s["status"] == "error"]
            ),
            "warnings": warning_count,
            "errors": error_count,
        }

        print(
            f"Data processing completed. Errors: {error_count}, Warnings: {warning_count}"
        )

        return result

    test_data = {
        "user": {"id": "user123", "name": "Test User"},
        "products": [
            {"id": "prod1", "name": "Laptop", "price": 1000, "category": "electronics"},
            {"id": "prod2", "name": "Book", "price": 25, "category": "books"},
        ],
        "orders": [
            {
                "order_id": "order123",
                "items": [
                    {"product_id": "prod1", "quantity": 1},
                    {"product_id": "prod2", "quantity": 2},
                ],
            }
        ],
    }

    result = massive_data_processor(test_data)
    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
