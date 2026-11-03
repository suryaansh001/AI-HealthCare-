const toast = document.querySelector("#toast");
let toastTimer;
let selectedMode = "low";
const simulationModes = {
  low: {
    label: "Low", mood: 8, sleep: 7.8, heartRate: 62, steps: 8240, stress: "Low",
    sleepScore: 86, activity: "Take a 15-minute walk", activityCopy: "A calm walk outside can reinforce the steady rhythm your signals are showing.",
    preview: "Low mode selected · ready to simulate a steady day"
  },
  medium: {
    label: "Medium", mood: 6, sleep: 6.1, heartRate: 76, steps: 4210, stress: "Moderate",
    sleepScore: 64, activity: "Listen to music and move", activityCopy: "Put on a song you like and take a gentle walk, sway, or stretch for a few minutes.",
    preview: "Medium mode selected · ready to simulate a day that needs attention"
  },
  high: {
    label: "High", mood: 3, sleep: 4.2, heartRate: 94, steps: 1380, stress: "Elevated",
    sleepScore: 42, activity: "Go somewhere supported", activityCopy: "Step outside or move to a shared space. Take a slow walk there, then contact someone in your support circle.",
    preview: "High mode selected · ready to simulate a day that needs extra support"
  }
};
function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3200);
}

document.querySelectorAll(".mode-option").forEach((button) => {
  button.addEventListener("click", () => {
    selectedMode = button.dataset.mode;
    document.querySelectorAll(".mode-option").forEach((option) => option.classList.toggle("selected", option === button));
    document.querySelector("#modePreviewText").textContent = simulationModes[selectedMode].preview;
  });
});

function setMetricContent(mode) {
  document.querySelector("#metricSleep").innerHTML = `${mode.sleepScore}<span>/100</span>`;
  document.querySelector("#sleepMeter").style.width = `${mode.sleepScore}%`;
  document.querySelector("#metricSleepNote").textContent = `${mode.sleep}h · ${mode.sleepScore >= 75 ? "consistent" : mode.sleepScore >= 55 ? "a little short" : "limited"}`;
  document.querySelector("#metricHeart").innerHTML = `${mode.heartRate}<span> bpm</span>`;
  document.querySelector("#metricHeartNote").textContent = mode.heartRate <= 72 ? "Within your personal range" : "Above your personal range";
  document.querySelector("#metricSteps").innerHTML = `${mode.steps.toLocaleString()}<span> steps</span>`;
  document.querySelector("#stepsMeter").style.width = `${Math.min(100, Math.round(mode.steps / 100))}%`;
  document.querySelector("#metricStepsNote").textContent = `${Math.round(mode.steps / 100)}% of your daily goal`;
  document.querySelector("#metricStress").innerHTML = `${mode.stress}<span> today</span>`;
  document.querySelector("#metricStressNote").textContent = `Simulated ${mode.label.toLowerCase()}-risk wearable pattern`;
}

function replayAnimation(selector, className = "is-animating") {
  const element = document.querySelector(selector);
  element.classList.remove(className);
  void element.offsetWidth;
  element.classList.add(className);
}

function setPlanActivity(title, copy) {
  document.querySelector("#planActivityTitle").textContent = title;
  document.querySelector("#planActivityCopy").textContent = copy;
}

function setStatusTheme(status) {
  const content = document.querySelector(".main-content");
  content.classList.remove("status-normal", "status-medium", "status-critical");
  content.classList.add(`status-${status}`);
}

function runModeSimulation() {
  const mode = simulationModes[selectedMode];
  const status = selectedMode === "high" ? "critical" : selectedMode === "medium" ? "medium" : "normal";
  setStatusTheme(status);
  setMetricContent(mode);
  setPlanActivity(mode.activity, mode.activityCopy);
  document.querySelectorAll(".metric-card").forEach((card, index) => {
    card.style.setProperty("--stagger", `${index * 80}ms`);
    card.classList.remove("signal-refresh");
    void card.offsetWidth;
    card.classList.add("signal-refresh");
  });
  const syntheticRisk = selectedMode === "low" ? 14 : selectedMode === "medium" ? 38 : 68;
  document.querySelector("#riskScore").textContent = syntheticRisk;
  const high = selectedMode === "high";
  const watchful = selectedMode === "medium";
  const label = high ? "HIGHER SUPPORT" : watchful ? "WATCHFUL" : "LOWER RISK";
  document.querySelector("#riskLabel").textContent = label;
  updateSimulationResult({
    mood: mode.mood, sleep: mode.sleep, heartRate: mode.heartRate, risk: syntheticRisk,
    isHighSupport: high, isWatchful: watchful, activity: mode.activity, activityCopy: mode.activityCopy,
    modeLabel: mode.label
  });
  document.querySelector("#simulationResult").hidden = false;
  replayAnimation(".score-ring", "score-refresh");
  replayAnimation("#simulationResult", "result-refresh");
  replayAnimation(".action-panel", "action-refresh");
  document.querySelector("#simulationResult").scrollIntoView({ behavior: "smooth", block: "center" });
  showToast(`${mode.label} mode simulated. Signals and support plan updated.`);
}
document.querySelector("#runSimulation").addEventListener("click", runModeSimulation);

const modal = document.querySelector("#checkinModal");
const openModal = () => { modal.hidden = false; document.querySelector("#moodRange").focus(); };
const closeModal = () => { modal.hidden = true; };
document.querySelector("#checkInButton").addEventListener("click", openModal);
document.querySelector("#closeModal").addEventListener("click", closeModal);
modal.addEventListener("click", (event) => { if (event.target === modal) closeModal(); });
document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !modal.hidden) closeModal(); });

const moodRange = document.querySelector("#moodRange");
const moodOutput = document.querySelector("#moodOutput");
moodRange.addEventListener("input", () => { moodOutput.textContent = `${moodRange.value} / 10`; });
document.querySelector("#checkinForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const sleep = Number(document.querySelector("#sleepInput").value);
  const mood = Number(moodRange.value);
  const heartRate = Number(document.querySelector("#heartRateInput").value);
  const risk = Math.max(8, Math.min(86, Math.round(45 - mood * 3 - sleep * 1.3 + Math.max(0, heartRate - 72) * 0.45 + Math.max(0, 6 - sleep) * 2)));
  const isHighSupport = risk >= 50;
  const isWatchful = risk >= 25;
  setStatusTheme(isHighSupport ? "critical" : isWatchful ? "medium" : "normal");
  document.querySelector("#riskScore").textContent = risk;
  document.querySelector("#riskLabel").textContent = isHighSupport ? "HIGHER SUPPORT" : isWatchful ? "WATCHFUL" : "LOWER RISK";
  updateSimulationResult({ mood, sleep, heartRate, risk, isHighSupport, isWatchful });
  closeModal();
  document.querySelector("#simulationResult").hidden = false;
  document.querySelector("#simulationResult").scrollIntoView({ behavior: "smooth", block: "center" });
  showToast("Check-in saved. Simulation results are ready below.");
});

function updateSimulationResult({ mood, sleep, heartRate, risk, isHighSupport, isWatchful, activity, activityCopy, modeLabel }) {
  const result = document.querySelector("#simulationResult");
  const moodQuality = mood >= 7 ? "steady" : mood >= 4 ? "mixed" : "strained";
  const sleepQuality = sleep >= 7 ? "supportive" : sleep >= 5.5 ? "limited" : "low";
  const heartQuality = heartRate <= 72 ? "within baseline" : "above baseline";
  document.querySelector("#resultTitle").textContent = modeLabel ? `${modeLabel} mode · ${isHighSupport ? "extra support" : isWatchful ? "needs attention" : "steady signals"}` : isHighSupport ? "Your readings suggest extra support" : isWatchful ? "A few signals need attention" : "Your readings look steady";
  document.querySelector("#resultSummary").textContent = isHighSupport ? "This pattern may be a good moment to bring someone into your plan." : isWatchful ? "A couple of changes moved your supportive score upward." : "Your current pattern looks steady.";
  document.querySelector("#resultDescription").textContent = `The sample model combined steadiness, ${sleep} hours of sleep, and a resting heart rate of ${heartRate} bpm. This is a simulation, not a diagnosis.`;
  document.querySelector("#resultIcon").textContent = isHighSupport ? "!" : isWatchful ? "↗" : "✓";
  document.querySelector("#resultIcon").className = `result-summary-icon ${isHighSupport ? "alert" : isWatchful ? "watch" : ""}`;
  document.querySelector("#resultMood").textContent = `${mood} / 10 · ${moodQuality}`;
  document.querySelector("#resultSleep").textContent = `${sleep}h · ${sleepQuality}`;
  document.querySelector("#resultHeart").textContent = `${heartRate} bpm · ${heartQuality}`;
  document.querySelector("#moodBar").style.width = `${mood * 10}%`;
  document.querySelector("#sleepBar").style.width = `${Math.min(100, sleep / 8 * 100)}%`;
  document.querySelector("#heartBar").style.width = `${Math.min(100, Math.max(25, 100 - Math.abs(heartRate - 64) * 2))}%`;
  const drivers = [];
  if (mood < 7) drivers.push(`Lower steadiness added ${Math.round((7 - mood) * 3)} points`);
  if (sleep < 7) drivers.push(`Shorter sleep added ${Math.round((7 - sleep) * 2)} points`);
  if (heartRate > 72) drivers.push(`Heart rate above baseline added ${Math.round((heartRate - 72) * .45)} points`);
  if (!drivers.length) drivers.push("Steady mood and restorative sleep reduced the simulated score");
  document.querySelector("#driverList").innerHTML = drivers.map((driver) => `<li><span>•</span>${driver}</li>`).join("");
  document.querySelector("#resultActivity").textContent = activity || (isHighSupport ? "Go somewhere supported" : isWatchful ? "Listen to music and move" : "Take a 15-minute walk");
  document.querySelector("#resultActivityCopy").textContent = activityCopy || (isHighSupport ? "Step outside or move to a shared space. Take a slow walk there, then contact someone in your support circle." : isWatchful ? "Put on a song you like and take a gentle walk, sway, or stretch for a few minutes." : "A calm walk outside can reinforce the steady rhythm your signals are showing.");
  setPlanActivity(activity || (isHighSupport ? "Go somewhere supported" : isWatchful ? "Listen to music and move" : "Take a 15-minute walk"), document.querySelector("#resultActivityCopy").textContent);
  document.querySelector("#notificationTitle").textContent = isHighSupport ? "Support notification is ready" : "No support notification needed";
  document.querySelector("#notificationCopy").textContent = isHighSupport ? "In a connected version, Anchor would ask for your confirmation before sending Jamie a WhatsApp or email check-in." : "Your settings are respected. We will only notify your circle when you choose to enable it and the pattern needs extra support.";
  document.querySelector("#notificationState").className = `notification-state ${isHighSupport ? "needs-support" : ""}`;
  result.dataset.score = risk;
}

document.querySelector("#activityButton").addEventListener("click", () => showToast("Activity started — 10 quiet minutes, just for you."));
document.querySelector("#resultAction").addEventListener("click", () => showToast("Added to your gentle plan. You can change this anytime."));
document.querySelectorAll("[data-activity]").forEach((button) => button.addEventListener("click", () => showToast(`${button.dataset.activity} added to your gentle plan.`)));
document.querySelector("#addCircle").addEventListener("click", () => showToast("Invite flow ready — WhatsApp and email connections will be added here."));
document.querySelector("#manageCircle").addEventListener("click", () => showToast("Circle settings opened. You choose who gets notified."));
document.querySelector("#helpButton").addEventListener("click", () => showToast("You’re not alone. Reach out to someone in your support circle."));
document.querySelector("#notificationButton").addEventListener("click", () => showToast("No urgent alerts. Two gentle reminders are waiting."));
document.querySelector("#insightDetails").addEventListener("click", () => document.querySelector("#signals").scrollIntoView({ behavior: "smooth" }));
document.querySelector("#riskExplain").addEventListener("click", () => showToast("Risk combines recent patterns, personal baseline changes, sleep, movement and check-ins."));
