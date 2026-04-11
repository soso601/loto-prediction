import { useState, useCallback, useEffect, useRef } from "react";

const TOTAL_COMBOS = 19_068_840;

// Contraintes disponibles
const CONSTRAINT_TYPES = [
  { id: "sum_range", label: "Somme des 5 numéros", icon: "Σ", desc: "Filtrer par la somme totale" },
  { id: "even_odd", label: "Répartition pair/impair", icon: "⚖", desc: "Nombre de pairs vs impairs" },
  { id: "decade", label: "Distribution par dizaine", icon: "📊", desc: "Numéros par tranche de 10" },
  { id: "consecutive", label: "Numéros consécutifs", icon: "🔗", desc: "Combien de suites autorisées" },
  { id: "amplitude", label: "Amplitude (max - min)", icon: "↔", desc: "Écart entre plus grand et plus petit" },
  { id: "numerology", label: "Somme numérologique", icon: "🔮", desc: "Réduction à un chiffre de la somme" },
  { id: "hot_cold", label: "Numéros chauds/froids", icon: "🌡", desc: "Exclure certains numéros" },
  { id: "chance", label: "Numéro Chance", icon: "⭐", desc: "Restreindre le numéro chance" },
];

// Estimations de réduction (approximatives pour l'affichage)
const REDUCTION_ESTIMATES = {
  sum_range: (min, max) => {
    const fullRange = 245 - 15;
    const selectedRange = max - min;
    return Math.max(0.05, Math.min(1, selectedRange / fullRange));
  },
  even_odd: (evens) => {
    const combos = { 0: 0.032, 1: 0.156, 2: 0.312, 3: 0.312, 4: 0.156, 5: 0.032 };
    return evens.reduce((acc, e) => acc + (combos[e] || 0), 0);
  },
  decade: () => 0.6,
  consecutive: (max) => max === 0 ? 0.35 : max === 1 ? 0.75 : 0.95,
  amplitude: (min, max) => Math.max(0.1, (max - min) / 48),
  numerology: (allowed) => allowed.length / 9,
  hot_cold: (excluded) => Math.pow((49 - excluded) / 49, 5),
  chance: (nums) => nums.length / 10,
};

function formatNumber(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(0) + "K";
  return n.toFixed(0);
}

function ConstraintCard({ type, config, onUpdate, onRemove }) {
  const renderInputs = () => {
    switch (type.id) {
      case "sum_range":
        return (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <label style={{ fontSize: 12, color: "var(--muted)" }}>
              Somme entre {config.min || 15} et {config.max || 245}
            </label>
            <input type="range" min={15} max={245} value={config.min || 80}
              onChange={e => onUpdate({ ...config, min: +e.target.value })}
              style={{ accentColor: "var(--accent)" }} />
            <input type="range" min={15} max={245} value={config.max || 180}
              onChange={e => onUpdate({ ...config, max: +e.target.value })}
              style={{ accentColor: "var(--accent)" }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
              <span style={{
                background: "var(--accent-bg)", padding: "2px 10px", borderRadius: 6
              }}>Min: {config.min || 80}</span>
              <span style={{
                background: "var(--accent-bg)", padding: "2px 10px", borderRadius: 6
              }}>Max: {config.max || 180}</span>
            </div>
          </div>
        );
      case "even_odd":
        return (
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {[0, 1, 2, 3, 4, 5].map(n => (
              <button key={n}
                onClick={() => {
                  const evens = config.evens || [2, 3];
                  const next = evens.includes(n) ? evens.filter(e => e !== n) : [...evens, n];
                  onUpdate({ ...config, evens: next });
                }}
                style={{
                  padding: "6px 14px", borderRadius: 8, border: "none", cursor: "pointer",
                  fontSize: 13, fontWeight: 600, transition: "all 0.2s",
                  background: (config.evens || [2, 3]).includes(n) ? "var(--accent)" : "var(--card-bg)",
                  color: (config.evens || [2, 3]).includes(n) ? "#fff" : "var(--text)",
                }}>
                {n}P/{5 - n}I
              </button>
            ))}
          </div>
        );
      case "consecutive":
        return (
          <div style={{ display: "flex", gap: 6 }}>
            {[0, 1, 2].map(n => (
              <button key={n}
                onClick={() => onUpdate({ ...config, max: n })}
                style={{
                  padding: "8px 18px", borderRadius: 8, border: "none", cursor: "pointer",
                  fontSize: 13, fontWeight: 600, transition: "all 0.2s",
                  background: (config.max ?? 1) === n ? "var(--accent)" : "var(--card-bg)",
                  color: (config.max ?? 1) === n ? "#fff" : "var(--text)",
                }}>
                {n === 0 ? "Aucun" : n === 1 ? "Max 1 paire" : "Max 2 paires"}
              </button>
            ))}
          </div>
        );
      case "amplitude":
        return (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <label style={{ fontSize: 12, color: "var(--muted)" }}>
              Amplitude entre {config.min || 15} et {config.max || 48}
            </label>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <input type="number" min={4} max={48} value={config.min || 15}
                onChange={e => onUpdate({ ...config, min: +e.target.value })}
                style={{
                  width: 60, padding: "6px 8px", borderRadius: 6,
                  border: "1px solid var(--border)", background: "var(--card-bg)",
                  color: "var(--text)", fontSize: 14
                }} />
              <span style={{ color: "var(--muted)" }}>→</span>
              <input type="number" min={4} max={48} value={config.max || 48}
                onChange={e => onUpdate({ ...config, max: +e.target.value })}
                style={{
                  width: 60, padding: "6px 8px", borderRadius: 6,
                  border: "1px solid var(--border)", background: "var(--card-bg)",
                  color: "var(--text)", fontSize: 14
                }} />
            </div>
          </div>
        );
      case "numerology":
        return (
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
            {[1, 2, 3, 4, 5, 6, 7, 8, 9].map(n => (
              <button key={n}
                onClick={() => {
                  const allowed = config.allowed || [1, 2, 3, 4, 5, 6, 7, 8, 9];
                  const next = allowed.includes(n) ? allowed.filter(e => e !== n) : [...allowed, n];
                  onUpdate({ ...config, allowed: next });
                }}
                style={{
                  width: 38, height: 38, borderRadius: "50%", border: "none", cursor: "pointer",
                  fontSize: 15, fontWeight: 700, transition: "all 0.2s",
                  background: (config.allowed || [1, 2, 3, 4, 5, 6, 7, 8, 9]).includes(n)
                    ? "var(--accent)" : "var(--card-bg)",
                  color: (config.allowed || [1, 2, 3, 4, 5, 6, 7, 8, 9]).includes(n)
                    ? "#fff" : "var(--text)",
                }}>
                {n}
              </button>
            ))}
          </div>
        );
      case "hot_cold":
        return (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <label style={{ fontSize: 12, color: "var(--muted)" }}>
              Numéros à exclure (cliquer pour toggler)
            </label>
            <div style={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
              {Array.from({ length: 49 }, (_, i) => i + 1).map(n => (
                <button key={n}
                  onClick={() => {
                    const excluded = config.excluded || [];
                    const next = excluded.includes(n)
                      ? excluded.filter(e => e !== n) : [...excluded, n];
                    onUpdate({ ...config, excluded: next });
                  }}
                  style={{
                    width: 32, height: 28, borderRadius: 4, border: "none", cursor: "pointer",
                    fontSize: 11, fontWeight: 600, transition: "all 0.15s",
                    background: (config.excluded || []).includes(n) ? "#e74c3c" : "var(--card-bg)",
                    color: (config.excluded || []).includes(n) ? "#fff" : "var(--text)",
                    opacity: (config.excluded || []).includes(n) ? 1 : 0.7,
                  }}>
                  {n}
                </button>
              ))}
            </div>
            <span style={{ fontSize: 11, color: "var(--muted)" }}>
              {(config.excluded || []).length} numéros exclus
            </span>
          </div>
        );
      case "chance":
        return (
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(n => (
              <button key={n}
                onClick={() => {
                  const nums = config.nums || [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
                  const next = nums.includes(n) ? nums.filter(e => e !== n) : [...nums, n];
                  onUpdate({ ...config, nums: next });
                }}
                style={{
                  width: 40, height: 40, borderRadius: "50%", border: "2px solid",
                  borderColor: (config.nums || [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).includes(n)
                    ? "var(--gold)" : "transparent",
                  cursor: "pointer", fontSize: 14, fontWeight: 700, transition: "all 0.2s",
                  background: (config.nums || [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).includes(n)
                    ? "var(--gold-bg)" : "var(--card-bg)",
                  color: (config.nums || [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).includes(n)
                    ? "var(--gold)" : "var(--text)",
                }}>
                {n}
              </button>
            ))}
          </div>
        );
      case "decade":
        return (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <label style={{ fontSize: 12, color: "var(--muted)" }}>
              Min/Max de numéros par dizaine
            </label>
            {["1-10", "11-20", "21-30", "31-40", "41-49"].map(range => (
              <div key={range} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 50, fontSize: 12, fontWeight: 600 }}>{range}</span>
                <select
                  value={(config[range] || {}).min ?? 0}
                  onChange={e => onUpdate({
                    ...config,
                    [range]: { ...(config[range] || {}), min: +e.target.value }
                  })}
                  style={{
                    padding: "3px 6px", borderRadius: 4, fontSize: 12,
                    border: "1px solid var(--border)", background: "var(--card-bg)",
                    color: "var(--text)"
                  }}>
                  {[0, 1, 2, 3, 4, 5].map(n => <option key={n} value={n}>≥{n}</option>)}
                </select>
                <select
                  value={(config[range] || {}).max ?? 5}
                  onChange={e => onUpdate({
                    ...config,
                    [range]: { ...(config[range] || {}), max: +e.target.value }
                  })}
                  style={{
                    padding: "3px 6px", borderRadius: 4, fontSize: 12,
                    border: "1px solid var(--border)", background: "var(--card-bg)",
                    color: "var(--text)"
                  }}>
                  {[0, 1, 2, 3, 4, 5].map(n => <option key={n} value={n}>≤{n}</option>)}
                </select>
              </div>
            ))}
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: 14,
      padding: 18,
      position: "relative",
      transition: "all 0.2s",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 22 }}>{type.icon}</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14 }}>{type.label}</div>
            <div style={{ fontSize: 11, color: "var(--muted)" }}>{type.desc}</div>
          </div>
        </div>
        <button onClick={onRemove} style={{
          background: "none", border: "none", color: "var(--muted)",
          cursor: "pointer", fontSize: 18, padding: 4
        }}>✕</button>
      </div>
      {renderInputs()}
    </div>
  );
}

// Chat IA
function ChatMessage({ role, content }) {
  return (
    <div style={{
      display: "flex",
      justifyContent: role === "user" ? "flex-end" : "flex-start",
      marginBottom: 10,
    }}>
      <div style={{
        maxWidth: "85%",
        padding: "10px 16px",
        borderRadius: role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
        background: role === "user" ? "var(--accent)" : "var(--surface)",
        color: role === "user" ? "#fff" : "var(--text)",
        fontSize: 13,
        lineHeight: 1.5,
        border: role === "user" ? "none" : "1px solid var(--border)",
        whiteSpace: "pre-wrap",
      }}>
        {content}
      </div>
    </div>
  );
}

export default function LotoAgent() {
  const [activeConstraints, setActiveConstraints] = useState([]);
  const [chatMessages, setChatMessages] = useState([
    {
      role: "assistant",
      content: `Salut Roukhaya ! 👋 Je suis ton assistant Loto.

Mon rôle : t'aider à réduire les 19 068 840 combinaisons possibles en appliquant des filtres statistiques intelligents.

🎯 Ajoute des contraintes dans le panneau de gauche, ou pose-moi des questions ici :
• "Quelle somme est la plus fréquente ?"
• "Combien de combinaisons avec 3 pairs ?"
• "Quels filtres recommandes-tu ?"

Plus tu ajoutes de contraintes, plus on réduit le champ !`
    }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showAddPanel, setShowAddPanel] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const calculateReduction = useCallback(() => {
    let factor = 1;
    for (const c of activeConstraints) {
      switch (c.type) {
        case "sum_range":
          factor *= REDUCTION_ESTIMATES.sum_range(c.config.min || 80, c.config.max || 180);
          break;
        case "even_odd":
          factor *= REDUCTION_ESTIMATES.even_odd(c.config.evens || [2, 3]);
          break;
        case "decade":
          factor *= REDUCTION_ESTIMATES.decade();
          break;
        case "consecutive":
          factor *= REDUCTION_ESTIMATES.consecutive(c.config.max ?? 1);
          break;
        case "amplitude":
          factor *= REDUCTION_ESTIMATES.amplitude(c.config.min || 15, c.config.max || 48);
          break;
        case "numerology":
          factor *= REDUCTION_ESTIMATES.numerology(c.config.allowed || [1, 2, 3, 4, 5, 6, 7, 8, 9]);
          break;
        case "hot_cold":
          factor *= REDUCTION_ESTIMATES.hot_cold((c.config.excluded || []).length);
          break;
        case "chance":
          factor *= REDUCTION_ESTIMATES.chance(c.config.nums || [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
          break;
      }
    }
    return Math.max(1, Math.round(TOTAL_COMBOS * factor));
  }, [activeConstraints]);

  const remaining = calculateReduction();
  const reductionPct = ((1 - remaining / TOTAL_COMBOS) * 100).toFixed(1);

  const addConstraint = (typeId) => {
    if (activeConstraints.find(c => c.type === typeId)) return;
    setActiveConstraints(prev => [...prev, { type: typeId, config: {} }]);
    setShowAddPanel(false);
  };

  const removeConstraint = (idx) => {
    setActiveConstraints(prev => prev.filter((_, i) => i !== idx));
  };

  const updateConstraint = (idx, config) => {
    setActiveConstraints(prev => prev.map((c, i) => i === idx ? { ...c, config } : c));
  };

  const buildSystemPrompt = () => {
    const constraintsSummary = activeConstraints.map(c => {
      const type = CONSTRAINT_TYPES.find(t => t.id === c.type);
      return `- ${type?.label}: ${JSON.stringify(c.config)}`;
    }).join("\n");

    return `Tu es un assistant expert en probabilités et statistiques du Loto français.
Le Loto français : 5 numéros parmi 49 (C(49,5) = 1 906 884) + 1 numéro Chance parmi 10. Total : 19 068 840 combinaisons.

Contraintes actives de l'utilisateur :
${constraintsSummary || "Aucune contrainte active"}

Combinaisons restantes estimées : ${formatNumber(remaining)} (réduction de ${reductionPct}%)

Ton rôle :
1. Répondre aux questions sur les statistiques du Loto
2. Suggérer des filtres pertinents pour réduire les combinaisons
3. Expliquer l'impact de chaque contrainte
4. Être honnête : le Loto est aléatoire, aucune stratégie ne garantit un gain
5. Aider à construire des grilles "statistiquement informées"

Réponds en français, de façon concise et utile. Utilise des émojis avec parcimonie.`;
  };

  const sendMessage = async () => {
    if (!chatInput.trim() || isLoading) return;
    const userMsg = chatInput.trim();
    setChatInput("");
    setChatMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setIsLoading(true);

    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: buildSystemPrompt(),
          messages: [
            ...chatMessages.filter(m => m.role !== "assistant" || chatMessages.indexOf(m) !== 0).map(m => ({
              role: m.role, content: m.content
            })),
            { role: "user", content: userMsg }
          ],
        }),
      });

      const data = await response.json();
      const text = data.content?.map(i => i.text || "").filter(Boolean).join("\n")
        || "Désolé, je n'ai pas pu générer une réponse.";
      setChatMessages(prev => [...prev, { role: "assistant", content: text }]);
    } catch (err) {
      setChatMessages(prev => [...prev, {
        role: "assistant",
        content: "⚠️ Erreur de connexion. Vérifie ta connexion et réessaie."
      }]);
    }
    setIsLoading(false);
  };

  return (
    <div style={{
      "--accent": "#6366f1", "--accent-bg": "rgba(99,102,241,0.12)",
      "--gold": "#f59e0b", "--gold-bg": "rgba(245,158,11,0.12)",
      "--surface": "rgba(255,255,255,0.03)", "--card-bg": "rgba(255,255,255,0.06)",
      "--border": "rgba(255,255,255,0.1)", "--text": "#e2e8f0",
      "--muted": "#94a3b8", "--bg": "#0f172a",
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      color: "var(--text)", background: "var(--bg)",
      height: "100vh", display: "flex", flexDirection: "column", overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        padding: "16px 24px",
        borderBottom: "1px solid var(--border)",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        flexShrink: 0,
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 20, fontWeight: 800, letterSpacing: -0.5 }}>
            🎰 Loto Agent
          </h1>
          <p style={{ margin: "2px 0 0", fontSize: 12, color: "var(--muted)" }}>
            Réduction intelligente des combinaisons
          </p>
        </div>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1 }}>
              Combinaisons restantes
            </div>
            <div style={{
              fontSize: 22, fontWeight: 800,
              background: "linear-gradient(135deg, var(--accent), var(--gold))",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>
              {formatNumber(remaining)}
            </div>
          </div>
          <div style={{
            width: 54, height: 54, borderRadius: "50%",
            background: `conic-gradient(var(--accent) ${reductionPct}%, var(--card-bg) 0)`,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <div style={{
              width: 42, height: 42, borderRadius: "50%", background: "var(--bg)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 11, fontWeight: 700,
            }}>
              {reductionPct}%
            </div>
          </div>
        </div>
      </div>

      {/* Main */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Left: Constraints */}
        <div style={{
          width: 380, borderRight: "1px solid var(--border)",
          display: "flex", flexDirection: "column", overflow: "hidden", flexShrink: 0,
        }}>
          <div style={{
            padding: "12px 18px",
            display: "flex", justifyContent: "space-between", alignItems: "center",
            borderBottom: "1px solid var(--border)",
          }}>
            <span style={{ fontSize: 13, fontWeight: 700 }}>
              Filtres actifs ({activeConstraints.length})
            </span>
            <button onClick={() => setShowAddPanel(!showAddPanel)} style={{
              background: "var(--accent)", color: "#fff", border: "none",
              borderRadius: 8, padding: "6px 14px", fontSize: 12, fontWeight: 700,
              cursor: "pointer", transition: "all 0.2s",
            }}>
              + Ajouter
            </button>
          </div>

          {/* Add panel */}
          {showAddPanel && (
            <div style={{
              padding: 12, borderBottom: "1px solid var(--border)",
              background: "rgba(99,102,241,0.05)",
              display: "flex", flexDirection: "column", gap: 6,
              maxHeight: 280, overflowY: "auto",
            }}>
              {CONSTRAINT_TYPES.filter(t => !activeConstraints.find(c => c.type === t.id)).map(t => (
                <button key={t.id} onClick={() => addConstraint(t.id)} style={{
                  display: "flex", alignItems: "center", gap: 10,
                  padding: "10px 14px", borderRadius: 10, border: "1px solid var(--border)",
                  background: "var(--surface)", color: "var(--text)",
                  cursor: "pointer", transition: "all 0.15s", textAlign: "left",
                }}>
                  <span style={{ fontSize: 20 }}>{t.icon}</span>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600 }}>{t.label}</div>
                    <div style={{ fontSize: 11, color: "var(--muted)" }}>{t.desc}</div>
                  </div>
                </button>
              ))}
            </div>
          )}

          {/* Constraints list */}
          <div style={{ flex: 1, overflowY: "auto", padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
            {activeConstraints.length === 0 && (
              <div style={{
                textAlign: "center", padding: 40, color: "var(--muted)", fontSize: 13,
              }}>
                <div style={{ fontSize: 40, marginBottom: 12 }}>🎯</div>
                Ajoute des filtres pour réduire<br />les 19M combinaisons
              </div>
            )}
            {activeConstraints.map((c, idx) => {
              const type = CONSTRAINT_TYPES.find(t => t.id === c.type);
              return (
                <ConstraintCard
                  key={c.type}
                  type={type}
                  config={c.config}
                  onUpdate={config => updateConstraint(idx, config)}
                  onRemove={() => removeConstraint(idx)}
                />
              );
            })}
          </div>
        </div>

        {/* Right: Chat */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div style={{ flex: 1, overflowY: "auto", padding: 20 }}>
            {chatMessages.map((msg, i) => (
              <ChatMessage key={i} role={msg.role} content={msg.content} />
            ))}
            {isLoading && (
              <div style={{
                display: "flex", gap: 6, padding: "10px 16px",
              }}>
                {[0, 1, 2].map(i => (
                  <div key={i} style={{
                    width: 8, height: 8, borderRadius: "50%", background: "var(--accent)",
                    animation: `pulse 1s ${i * 0.15}s infinite`,
                    opacity: 0.6,
                  }} />
                ))}
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <div style={{
            padding: "12px 20px", borderTop: "1px solid var(--border)",
            display: "flex", gap: 10,
          }}>
            <input
              type="text"
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && sendMessage()}
              placeholder="Pose une question sur les stats du Loto..."
              style={{
                flex: 1, padding: "12px 18px", borderRadius: 12,
                border: "1px solid var(--border)", background: "var(--surface)",
                color: "var(--text)", fontSize: 13, outline: "none",
              }}
            />
            <button onClick={sendMessage} disabled={isLoading} style={{
              background: "var(--accent)", color: "#fff", border: "none",
              borderRadius: 12, padding: "12px 24px", fontSize: 13,
              fontWeight: 700, cursor: isLoading ? "wait" : "pointer",
              opacity: isLoading ? 0.6 : 1, transition: "all 0.2s",
            }}>
              Envoyer
            </button>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { transform: scale(1); opacity: 0.4; }
          50% { transform: scale(1.3); opacity: 1; }
        }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
      `}</style>
    </div>
  );
}
