import json
from datetime import datetime
from backend.persistence.database import SessionLocal, SyllabusItem, LearningLog

class SyllabusManager:
    def __init__(self):
        pass

    def add_item(self, title, objective, target_state, order):
        db = SessionLocal()
        new_item = SyllabusItem(
            title=title,
            objective=objective,
            target_state_json=json.dumps(target_state),
            order=order
        )
        db.add(new_item)
        db.commit()
        db.close()

    def get_next_item(self):
        db = SessionLocal()
        item = db.query(SyllabusItem).filter(SyllabusItem.is_completed == False).order_by(SyllabusItem.order).first()
        db.close()
        return item

    def mark_completed(self, item_id):
        db = SessionLocal()
        item = db.query(SyllabusItem).filter(SyllabusItem.id == item_id).first()
        if item:
            item.is_completed = True
            item.completed_at = datetime.utcnow()
            db.commit()
        db.close()

    def list_items(self):
        """List all syllabus items (for CLI/UI)."""
        db = SessionLocal()
        items = db.query(SyllabusItem).order_by(SyllabusItem.order).all()
        db.close()
        return [
            {
                "id": i.id,
                "title": i.title,
                "objective": i.objective,
                "target_state_json": i.target_state_json,
                "order": i.order,
                "is_completed": i.is_completed,
            }
            for i in items
        ]

    def log_attempt(self, item_id, rule_id, success, error=None):
        db = SessionLocal()
        log = LearningLog(
            syllabus_item_id=item_id,
            rule_id=None if rule_id == "N/A" else rule_id,
            success=success,
            error_message=error
        )
        db.add(log)
        db.commit()
        db.close()

syllabus_manager = SyllabusManager()
