# Feature Extraction & Scoring

## Overview

The system extracts features across multiple dimensions to understand email importance and recommended actions. Each dimension contributes to the overall scoring.

## Scoring Dimensions

### 1. People Score

Who sent the email and who's involved matters significantly.

```python
@dataclass
class PeopleFeatures:
    # Sender analysis
    sender_email: str
    sender_domain: str
    sender_org_level: int          # 0=external, 1=peer, 2=manager, 3=executive
    sender_department: str
    sender_historical_importance: float  # Based on past interaction patterns

    # Relationship metrics
    emails_from_sender_30d: int    # Volume from this sender
    reply_rate_to_sender: float    # How often user replies to them
    avg_response_time_to_sender: float  # Typical response speed
    last_interaction_days: int     # Recency

    # Recipients analysis
    recipient_count: int
    cc_count: int
    is_direct_to: bool             # User in To: vs CC:
    includes_user_manager: bool
    includes_executives: bool

def compute_people_score(features: PeopleFeatures) -> float:
    """
    Score 0-1 based on people involved.

    Higher scores for:
    - Direct manager or executive
    - Frequent correspondents with high reply rates
    - Direct recipient (not CC)
    """
    score = 0.0

    # Organizational hierarchy weight
    hierarchy_weights = {0: 0.3, 1: 0.5, 2: 0.8, 3: 1.0}
    score += hierarchy_weights.get(features.sender_org_level, 0.3) * 0.3

    # Relationship strength
    if features.reply_rate_to_sender > 0.7:
        score += 0.2
    elif features.reply_rate_to_sender > 0.4:
        score += 0.1

    # Direct recipient bonus
    if features.is_direct_to:
        score += 0.2

    # Recency penalty for cold contacts
    if features.last_interaction_days > 90:
        score *= 0.8

    # Executive involvement
    if features.includes_executives:
        score += 0.2

    return min(score, 1.0)
```

### 2. Project Score

Is this email related to active projects?

```python
@dataclass
class ProjectFeatures:
    # Detected projects
    mentioned_projects: List[str]  # Extracted project names
    project_keywords: List[str]    # Related terms found

    # Project metadata (from lookup)
    project_priority: str          # high/medium/low
    project_deadline_days: int     # Days until deadline
    user_role_in_project: str      # owner/contributor/stakeholder
    project_status: str            # active/planning/blocked/completed

    # Historical
    project_email_volume: int      # Recent email activity
    project_response_pattern: float  # How urgently user handles this project

def extract_projects(email_text: str, known_projects: List[dict]) -> ProjectFeatures:
    """
    Extract project references from email content.

    Methods:
    1. Direct mention matching against known project names
    2. Code name detection (capitalized acronyms)
    3. Keyword clustering (ML-based)
    """
    mentioned = []

    # Direct matching
    for project in known_projects:
        if project['name'].lower() in email_text.lower():
            mentioned.append(project['name'])
        for alias in project.get('aliases', []):
            if alias.lower() in email_text.lower():
                mentioned.append(project['name'])

    # Acronym detection
    acronym_pattern = r'\b[A-Z]{2,6}\b'
    potential_acronyms = re.findall(acronym_pattern, email_text)

    return ProjectFeatures(
        mentioned_projects=mentioned,
        project_keywords=potential_acronyms,
        # ... lookup remaining fields
    )

def compute_project_score(features: ProjectFeatures) -> float:
    """
    Score 0-1 based on project relevance.
    """
    if not features.mentioned_projects:
        return 0.2  # Base score for non-project emails

    score = 0.5  # Base for project-related

    # Priority boost
    priority_boost = {'high': 0.3, 'medium': 0.15, 'low': 0.05}
    score += priority_boost.get(features.project_priority, 0)

    # Deadline urgency
    if features.project_deadline_days < 7:
        score += 0.2
    elif features.project_deadline_days < 30:
        score += 0.1

    # Role in project
    role_boost = {'owner': 0.2, 'contributor': 0.1, 'stakeholder': 0.05}
    score += role_boost.get(features.user_role_in_project, 0)

    return min(score, 1.0)
```

### 3. Topic Score

What is this email about? Topic classification.

```python
@dataclass
class TopicFeatures:
    # Topic model output
    primary_topic: str
    topic_distribution: Dict[str, float]  # {topic: probability}

    # Content classification
    is_meeting_request: bool
    is_status_update: bool
    is_question: bool
    is_fyi_only: bool
    is_action_request: bool
    is_decision_needed: bool
    is_escalation: bool

    # Sentiment
    sentiment_score: float  # -1 to 1
    urgency_language: float # 0 to 1, presence of urgent terms

# Pre-defined topic categories
TOPIC_CATEGORIES = [
    'meeting_scheduling',
    'project_update',
    'task_assignment',
    'information_sharing',
    'decision_request',
    'problem_report',
    'follow_up',
    'social_administrative',
    'external_communication',
    'legal_compliance',
]

def classify_topic(email: dict) -> TopicFeatures:
    """
    Classify email into topic categories.

    Uses:
    1. Rule-based detection for common patterns
    2. LDA topic model for general categorization
    3. Transformer classifier for action classification
    """
    text = f"{email['subject']} {email['body']}"

    # Rule-based detection
    is_meeting = bool(re.search(r'meeting|calendar|schedule|availability', text, re.I))
    is_question = '?' in text or bool(re.search(r'\b(what|when|where|why|how|can you|could you)\b', text, re.I))
    is_action = bool(re.search(r'please|action required|need you to|can you', text, re.I))
    is_urgent = bool(re.search(r'urgent|asap|immediately|critical|deadline', text, re.I))

    # Topic model (pre-trained LDA)
    topic_dist = topic_model.transform(vectorizer.transform([text]))[0]
    primary_topic = TOPIC_CATEGORIES[topic_dist.argmax()]

    return TopicFeatures(
        primary_topic=primary_topic,
        topic_distribution=dict(zip(TOPIC_CATEGORIES, topic_dist)),
        is_meeting_request=is_meeting,
        is_question=is_question,
        is_action_request=is_action,
        urgency_language=is_urgent * 1.0,
        # ... other fields
    )

def compute_topic_score(features: TopicFeatures) -> float:
    """
    Score based on topic importance.

    Topics requiring action score higher.
    """
    # Base score by topic type
    topic_weights = {
        'decision_request': 0.9,
        'problem_report': 0.85,
        'task_assignment': 0.8,
        'follow_up': 0.7,
        'meeting_scheduling': 0.6,
        'project_update': 0.5,
        'information_sharing': 0.3,
        'social_administrative': 0.2,
    }

    score = topic_weights.get(features.primary_topic, 0.5)

    # Modifiers
    if features.is_question:
        score += 0.1
    if features.is_action_request:
        score += 0.15
    if features.is_decision_needed:
        score += 0.2
    if features.urgency_language > 0.5:
        score += 0.15

    return min(score, 1.0)
```

### 4. Task Score

Should this become a tracked task?

```python
@dataclass
class TaskFeatures:
    # Task indicators
    has_deadline: bool
    deadline_date: Optional[datetime]
    has_deliverable: bool
    deliverable_description: str

    # Assignment
    is_assigned_to_user: bool
    assigned_by: str
    assignment_confidence: float

    # Task complexity
    estimated_effort: str  # quick/medium/substantial
    requires_others: bool
    is_blocker_for_others: bool

    # Extracted action items
    action_items: List[str]

def extract_tasks(email: dict) -> TaskFeatures:
    """
    Extract potential tasks from email.

    Looks for:
    - Explicit assignments ("Can you...", "Please...")
    - Deadlines ("by Friday", "EOD", specific dates)
    - Deliverables ("send me", "prepare", "review")
    """
    text = f"{email['subject']} {email['body']}"

    # Deadline extraction
    deadline_patterns = [
        r'by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
        r'by\s+(end of day|EOD|COB|end of week|EOW)',
        r'by\s+(\d{1,2}/\d{1,2})',
        r'due\s+(monday|tuesday|wednesday|thursday|friday)',
        r'deadline[:\s]+([^.]+)',
    ]

    deadline = None
    for pattern in deadline_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            deadline = parse_deadline(match.group(1))
            break

    # Action item extraction
    action_patterns = [
        r'(?:please|can you|could you|need you to)\s+([^.?!]+)',
        r'action\s*(?:item|required)[:\s]+([^.]+)',
        r'(?:^|\n)\s*[-•]\s*([^.\n]+(?:you|your)[^.\n]+)',
    ]

    action_items = []
    for pattern in action_patterns:
        matches = re.findall(pattern, text, re.I | re.M)
        action_items.extend(matches)

    return TaskFeatures(
        has_deadline=deadline is not None,
        deadline_date=deadline,
        action_items=action_items[:5],  # Top 5
        # ... other fields
    )

def compute_task_score(features: TaskFeatures) -> float:
    """
    Score likelihood this should become a task.
    """
    if not features.action_items and not features.has_deadline:
        return 0.1

    score = 0.3  # Base for having some task indicators

    if features.has_deadline:
        score += 0.3
        # Urgency by deadline proximity
        if features.deadline_date:
            days_until = (features.deadline_date - datetime.now()).days
            if days_until < 1:
                score += 0.3
            elif days_until < 3:
                score += 0.2
            elif days_until < 7:
                score += 0.1

    if features.is_assigned_to_user:
        score += 0.2

    if features.has_deliverable:
        score += 0.15

    if features.is_blocker_for_others:
        score += 0.2

    return min(score, 1.0)
```

### 5. Action Score

What action should be taken?

```python
@dataclass
class ActionFeatures:
    # Response indicators
    expects_reply: bool
    reply_urgency: str  # immediate/soon/eventually/none

    # Forward indicators
    should_delegate: bool
    suggested_delegate: Optional[str]

    # Archive indicators
    is_informational_only: bool
    is_automated: bool
    is_newsletter: bool

    # Historical patterns
    similar_emails_action: str  # What user typically does
    sender_typical_response: str

def compute_action_scores(
    email: dict,
    people: PeopleFeatures,
    topic: TopicFeatures,
    task: TaskFeatures
) -> Dict[str, float]:
    """
    Compute probability distribution over actions.
    """
    scores = {
        'reply_now': 0.0,
        'reply_later': 0.0,
        'forward': 0.0,
        'archive': 0.0,
        'delete': 0.0,
    }

    # Reply likelihood (5-class action space)
    if topic.is_question or topic.is_action_request:
        if people.sender_org_level >= 2:  # Manager or above
            scores['reply_now'] += 0.4
        else:
            scores['reply_later'] += 0.3

    # Forward likelihood
    if task.has_deadline or len(task.action_items) > 0:
        scores['forward'] += 0.2  # May need to delegate

    # Archive
    if topic.is_fyi_only and not topic.is_action_request:
        scores['archive'] += 0.4

    # Delete (automated, newsletters)
    if is_automated_email(email) or is_newsletter(email):
        scores['delete'] += 0.5

    # Normalize to probabilities
    total = sum(scores.values()) + 0.01
    return {k: v/total for k, v in scores.items()}
```

## Combined Priority Score

```python
def compute_overall_priority(
    people_score: float,
    project_score: float,
    topic_score: float,
    task_score: float,
) -> float:
    """
    Weighted combination of all scores.

    Weights can be learned or configured per user.
    """
    weights = {
        'people': 0.30,
        'project': 0.25,
        'topic': 0.25,
        'task': 0.20,
    }

    priority = (
        weights['people'] * people_score +
        weights['project'] * project_score +
        weights['topic'] * topic_score +
        weights['task'] * task_score
    )

    return priority
```

## Feature Vector Assembly

```python
def build_feature_vector(email: dict, user_context: dict) -> np.array:
    """
    Build complete feature vector for RL agent.

    Returns concatenated vector of all features.
    """
    # Text embeddings (using sentence transformers)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    subject_emb = embedder.encode(email['subject'])  # 384-dim
    body_emb = embedder.encode(email['body'][:1000])  # 384-dim, truncated

    # Extract structured features
    people = extract_people_features(email, user_context)
    project = extract_projects(email['body'], user_context['projects'])
    topic = classify_topic(email)
    task = extract_tasks(email)

    # Numerical features
    numerical = np.array([
        compute_people_score(people),
        compute_project_score(project),
        compute_topic_score(topic),
        compute_task_score(task),
        people.sender_org_level / 3.0,
        people.reply_rate_to_sender,
        len(email['body']) / 10000.0,  # Normalized length
        1.0 if topic.is_question else 0.0,
        1.0 if topic.is_action_request else 0.0,
        topic.urgency_language,
        # ... more features
    ])

    # Concatenate all
    return np.concatenate([
        subject_emb,
        body_emb,
        numerical,
    ])
```

## Context Requirements

For optimal prediction, the system needs context about:

```python
@dataclass
class UserContext:
    # User profile
    user_email: str
    user_department: str
    user_role: str
    user_manager: str

    # Active projects
    projects: List[dict]  # {name, priority, deadline, role}

    # Contact graph
    frequent_contacts: Dict[str, float]  # email -> importance
    org_chart: Dict[str, int]  # email -> level

    # Historical patterns
    response_time_by_sender: Dict[str, float]
    reply_rate_by_topic: Dict[str, float]
    typical_daily_email_volume: int

    # Current state
    pending_tasks: List[dict]
    calendar_events_today: List[dict]
    current_focus_project: Optional[str]
```

## Next Steps

See [Training Guide](./04-training.md) for how to train the model using these features.
