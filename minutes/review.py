from minutes.models import ActionItem


def apply_action_edits(actions: list[ActionItem], rows: list[dict]) -> list[ActionItem]:
    if len(actions) != len(rows):
        raise ValueError("Action count changed; regenerate the editor")
    updated = []
    for action, row in zip(actions, rows):
        data = action.model_dump()
        responsible = (row.get("Responsible") or "").strip() or None
        if responsible != action.responsible:
            data["responsible_speaker"] = None
        data.update(
            responsible=responsible,
            task=row["Task"],
            deadline=row["Deadline"] or None,
            status=row["Status"],
        )
        if data["deadline"] != action.deadline:
            data["deadline_text"] = str(data["deadline"]) if data["deadline"] else None
        updated.append(ActionItem.model_validate(data))
    return updated
