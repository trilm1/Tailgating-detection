import cv2


def annotate_frame(frame, observations, config, t_s, events, banner):
    threshold = config["threshold"]
    start, end = (tuple(round(v) for v in threshold[k]) for k in ("start", "end"))
    cv2.line(frame, start, end, (0, 230, 255), 2)
    mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx * dx + dy * dy)**0.5
    sign = threshold["in_side"]
    tip = (round(mid[0] - sign * dy / length * 45), round(mid[1] + sign * dx / length * 45))
    cv2.arrowedLine(frame, mid, tip, (0, 230, 255), 2)
    cv2.putText(frame, "IN", tip, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 230, 255), 2)
    for obs in observations:
        x1, y1, x2, y2 = (round(v) for v in obs.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 230, 100), 2)
        cv2.putText(frame, f"ID {obs.track_id}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (100, 230, 100), 2)
        cv2.circle(frame, ((x1 + x2)//2, y2), 3, (0, 230, 255), -1)
    active = [g for g in events.grants
              if g.t_unlock_s - events.lead_s <= t_s <= g.t_relock_s + events.grace_s]
    counts = " | ".join(f"{g.grant_id}: {len(events.crossings[g.grant_id])}" for g in active)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (25, 25, 25), -1)
    cv2.putText(frame, f"{t_s:.2f}s   Entries per grant: {counts or 'no active grant'}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    cv2.putText(frame, f"Unauthorised entries: {len(events.unauthorised)}", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1)
    if banner:
        cv2.rectangle(frame, (0, 60), (frame.shape[1], 96), (20, 20, 190), -1)
        cv2.putText(frame, banner, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    return frame
