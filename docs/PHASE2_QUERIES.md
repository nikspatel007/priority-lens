# Phase 2 Feature Queries

Run these queries against `gmail_twoyrs` database to analyze your email patterns.

## Connection
```bash
psql postgresql://postgres:postgres@localhost:5433/gmail_twoyrs
```

---

## 1. Priority Inbox
Real people + important service emails, ranked by priority.

```sql
SELECT e.subject, e.from_email, e.date_parsed,
       ef.relationship_strength,
       ef.urgency_score,
       ef.service_importance,
       ef.is_service_email
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE e.action = 'PENDING'
  AND (ef.is_service_email = FALSE
       OR ef.service_importance >= 0.5)
ORDER BY (ef.relationship_strength + ef.urgency_score +
          COALESCE(ef.service_importance, 0)) DESC
LIMIT 30;
```

---

## 2. Important Service Emails Only
Transactions, alerts, security notifications.

```sql
SELECT e.subject, e.from_email, e.date_parsed,
       ef.service_importance, ef.service_type
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE ef.is_service_email = TRUE
  AND ef.service_importance >= 0.7
ORDER BY e.date_parsed DESC
LIMIT 50;
```

---

## 3. Unsubscribe Candidates
High volume, low importance, never engaged.

```sql
SELECT e.from_email,
       COUNT(*) as email_count,
       AVG(ef.service_importance) as avg_importance,
       MAX(ef.service_type) as type,
       MAX(ef.has_unsubscribe_link::int) as has_unsub
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE ef.is_service_email = TRUE
  AND ef.service_importance < 0.4
  AND ef.user_replied_to_sender_rate = 0
GROUP BY e.from_email
HAVING COUNT(*) > 10
ORDER BY email_count DESC
LIMIT 30;
```

---

## 4. Chase Breakdown by Importance
See how Chase emails split by importance level.

```sql
SELECT
    CASE
        WHEN ef.service_importance >= 0.7 THEN 'HIGH - Action needed'
        WHEN ef.service_importance >= 0.4 THEN 'MEDIUM - FYI'
        ELSE 'LOW - Marketing'
    END as importance_level,
    COUNT(*) as cnt,
    ROUND(AVG(ef.service_importance)::numeric, 2) as avg_score
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE LOWER(e.from_email) LIKE '%chase%'
GROUP BY importance_level
ORDER BY avg_score DESC;
```

---

## 5. Top Relationships
Real people you engage with most.

```sql
SELECT e.from_email,
       ef.relationship_strength,
       ef.emails_from_sender_all as total_emails,
       ef.user_replied_to_sender_rate as reply_rate,
       ef.days_since_last_interaction as days_ago
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE ef.is_service_email = FALSE
  AND ef.relationship_strength > 0.3
GROUP BY e.from_email, ef.relationship_strength,
         ef.emails_from_sender_all, ef.user_replied_to_sender_rate,
         ef.days_since_last_interaction
ORDER BY ef.relationship_strength DESC
LIMIT 20;
```

---

## 6. Neglected Relationships
People you used to engage with but haven't recently.

```sql
SELECT e.from_email,
       ef.relationship_strength,
       ef.days_since_last_interaction,
       ef.user_replied_to_sender_rate,
       ef.emails_from_sender_all
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE ef.relationship_strength > 0.4
  AND ef.days_since_last_interaction > 30
  AND ef.is_service_email = FALSE
GROUP BY e.from_email, ef.relationship_strength,
         ef.days_since_last_interaction, ef.user_replied_to_sender_rate,
         ef.emails_from_sender_all
ORDER BY ef.relationship_strength DESC
LIMIT 20;
```

---

## 7. Service Email Summary by Sender
Volume and importance breakdown per sender.

```sql
SELECT e.from_email,
       COUNT(*) as total,
       ROUND(AVG(ef.service_importance)::numeric, 2) as avg_importance,
       SUM(CASE WHEN ef.service_importance >= 0.7 THEN 1 ELSE 0 END) as high,
       SUM(CASE WHEN ef.service_importance < 0.4 THEN 1 ELSE 0 END) as low,
       MAX(ef.service_type) as type
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE ef.is_service_email = TRUE
GROUP BY e.from_email
HAVING COUNT(*) > 20
ORDER BY total DESC
LIMIT 30;
```

---

## 8. Time Analysis
When do important emails arrive?

```sql
SELECT ef.time_bucket,
       COUNT(*) as total,
       SUM(CASE WHEN ef.is_service_email = FALSE THEN 1 ELSE 0 END) as from_people,
       SUM(CASE WHEN ef.service_importance >= 0.7 THEN 1 ELSE 0 END) as important_service
FROM email_features ef
GROUP BY ef.time_bucket
ORDER BY total DESC;
```

---

## 9. Amazon Breakdown
Example of segmented sender analysis.

```sql
SELECT e.from_email,
       COUNT(*) as cnt,
       ROUND(AVG(ef.service_importance)::numeric, 2) as avg_importance
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE LOWER(e.from_email) LIKE '%amazon%'
GROUP BY e.from_email
ORDER BY cnt DESC;
```

---

## 10. Emails That Actually Matter
Combined priority score ranking.

```sql
SELECT e.subject, e.from_email, e.date_parsed,
       ROUND((
           ef.relationship_strength * 0.4 +
           ef.urgency_score * 0.3 +
           CASE WHEN ef.is_service_email THEN ef.service_importance * 0.3
                ELSE 0.3 END
       )::numeric, 2) as priority_score,
       ef.is_service_email,
       ef.service_importance
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE e.action = 'PENDING'
ORDER BY priority_score DESC
LIMIT 30;
```
