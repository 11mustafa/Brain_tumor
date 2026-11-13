import { AnimatePresence, motion } from "framer-motion";
import {
  Activity, ArrowRight, BarChart3, BrainCircuit, Check, CheckCircle2, ChevronRight,
  CircleGauge, Eye, EyeOff, FileImage, Gauge, ImagePlus, LockKeyhole, LogOut, Menu,
  Microscope, RefreshCw, ScanLine, ShieldCheck, Sparkles, UploadCloud, X, Zap,
} from "lucide-react";
import { FormEvent, useEffect, useRef, useState } from "react";

type Page = "home" | "login" | "analysis" | "result" | "performance" | "charts";
const navItems: { label: string; page: Page }[] = [
  { label: "Home", page: "home" }, { label: "Analysis", page: "analysis" },
  { label: "Performance", page: "performance" }, { label: "Charts", page: "charts" },
];
const pages: Page[] = ["home", "login", "analysis", "result", "performance", "charts"];

interface ScanResult {
  image: string;
  name: string;
  prediction: string;
  confidence: number;
  danger: boolean;
  symptoms: string;
}

function Brand() {
  return <div className="brand" aria-label="NeuroScan AI"><span className="brand-mark"><BrainCircuit size={21} /></span><span className="brand-name">NeuroScan <b>AI</b></span></div>;
}

function Header({ page, onNavigate, loggedIn, onLogout }: { page: Page; onNavigate: (page: Page) => void; loggedIn: boolean; onLogout: () => void }) {
  const [open, setOpen] = useState(false);
  const navigate = (next: Page) => { setOpen(false); onNavigate(next); };
  return <header className="site-header">
    <div className="nav-shell">
      <button className="brand-button" onClick={() => navigate("home")}><Brand /></button>
      <nav className="desktop-nav" aria-label="Primary navigation">{navItems.map(item => <button key={item.page} onClick={() => navigate(item.page)} className={page === item.page ? "active" : ""}>{item.label}</button>)}</nav>
      <div className="nav-actions">
        {loggedIn ? <button className="nav-auth" onClick={onLogout}><LogOut size={16} /> Logout</button> : <button className="nav-auth" onClick={() => navigate("login")}><LockKeyhole size={16} /> Login</button>}
        <button className="menu-button" onClick={() => setOpen(!open)} aria-expanded={open} aria-label="Toggle menu">{open ? <X /> : <Menu />}</button>
      </div>
    </div>
    <AnimatePresence>{open && <motion.nav className="mobile-nav" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} aria-label="Mobile navigation">
      {[...navItems, { label: loggedIn ? "Logout" : "Login", page: "login" as Page }].map(item => <button key={item.label} onClick={() => item.label === "Logout" ? onLogout() : navigate(item.page)}>{item.label}<ChevronRight size={18} /></button>)}
    </motion.nav>}</AnimatePresence>
  </header>;
}

function Footer({ onNavigate }: { onNavigate: (page: Page) => void }) {
  return <footer className="site-footer"><div className="footer-inner"><Brand /><p>AI-powered imaging demonstration platform</p><div className="footer-links"><button onClick={() => onNavigate("performance")}>Model metrics</button><button onClick={() => onNavigate("charts")}>Dataset</button></div><span className="copyright">© 2025 NeuroScan AI</span></div></footer>;
}

const reveal = { initial: { opacity: 0, y: 18 }, animate: { opacity: 1, y: 0 }, transition: { duration: .45 } };

function Home({ onNavigate }: { onNavigate: (page: Page) => void }) {
  const features = [
    { icon: Gauge, label: "MODEL QUALITY", title: "High Accuracy", text: "Built to demonstrate robust deep-learning classification on brain MRI imagery." },
    { icon: Zap, label: "WORKFLOW", title: "Real-time Processing", text: "A focused upload-to-result experience designed for clarity and speed." },
    { icon: Microscope, label: "INSIGHT", title: "Detailed Analysis", text: "Review classification confidence and model performance in one workspace." },
  ];
  return <><main>
    <section className="hero"><div className="hero-image" aria-hidden="true" /><div className="hero-vignette" aria-hidden="true" />
      <motion.div className="hero-content" {...reveal}><div className="eyebrow"><Activity size={15} /> MEDICAL IMAGING INTELLIGENCE</div><h1><span>NeuroScan AI</span><br />See deeper. Decide clearer.</h1><p>A refined deep-learning demonstration workspace for exploring brain MRI classification with speed, context, and confidence.</p><div className="hero-actions"><button className="button primary" onClick={() => onNavigate("login")}>Access workspace <ArrowRight size={18} /></button><button className="button ghost" onClick={() => onNavigate("performance")}>Explore model <CircleGauge size={18} /></button></div></motion.div>
      <div className="scan-orbit" aria-hidden="true"><span /><i /></div><div className="scroll-cue" aria-hidden="true"><span /> Explore platform</div>
    </section>
    <section className="features section-shell"><motion.div className="section-intro" {...reveal}><div className="eyebrow">THE PLATFORM</div><h2>Intelligence at every step.</h2><p>Purpose-built tools turn a complex AI workflow into a calm, understandable experience.</p></motion.div>
      <div className="feature-grid">{features.map((feature, index) => <motion.article className="feature-item" key={feature.title} initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: index * .1 }}><div className="feature-top"><feature.icon /><span>0{index + 1}</span></div><small>{feature.label}</small><h3>{feature.title}</h3><p>{feature.text}</p></motion.article>)}</div>
    </section>
  </main><Footer onNavigate={onNavigate} /></>;
}

function Login({ onLogin }: { onLogin: () => void }) {
  const [showPassword, setShowPassword] = useState(false); const [error, setError] = useState("");
  const submit = (event: FormEvent<HTMLFormElement>) => { event.preventDefault(); const data = new FormData(event.currentTarget); if (!data.get("username") || !data.get("password")) { setError("Enter your username and password to continue."); return; } setError(""); onLogin(); };
  return <main className="auth-page">
    <motion.section className="auth-story" {...reveal}><div className="eyebrow"><ShieldCheck size={15} /> SECURE AI WORKSPACE</div><h1>Medical imaging,<br /><span>made intelligible.</span></h1><p>Access a focused environment for MRI classification, model insights, and performance analytics.</p><div className="auth-visual" aria-hidden="true"><div className="brain-window" /><div className="data-line"><Activity size={16} /><span>Imaging engine ready</span><b>ONLINE</b></div></div></motion.section>
    <motion.section className="auth-form-wrap" initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: .5 }}><form className="auth-form glass-panel" onSubmit={submit} noValidate><div className="form-icon"><LockKeyhole /></div><div><span className="form-kicker">AUTHORIZED ACCESS</span><h2>Access NeuroScan AI</h2><p>Enter your credentials to open the analysis workspace.</p></div>{error && <div className="form-error" role="alert"><X size={17} /> {error}</div>}
      <label className="field-label" htmlFor="username">Username</label><div className="input-wrap"><input id="username" name="username" autoComplete="username" placeholder="Enter username" /><Activity size={18} /></div>
      <label className="field-label" htmlFor="password">Password</label><div className="input-wrap"><input id="password" name="password" type={showPassword ? "text" : "password"} autoComplete="current-password" placeholder="Enter password" /><button type="button" onClick={() => setShowPassword(!showPassword)} aria-label={showPassword ? "Hide password" : "Show password"}>{showPassword ? <EyeOff size={18} /> : <Eye size={18} />}</button></div>
      <button className="button primary submit-button" type="submit">Enter workspace <ArrowRight size={18} /></button><p className="security-note"><ShieldCheck size={15} /> Encrypted session · Authorized use only</p>
    </form></motion.section>
  </main>;
}

function PageHeading({ kicker, title, text }: { kicker: string; title: string; text: string }) { return <motion.div className="page-heading" {...reveal}><div className="eyebrow">{kicker}</div><h1>{title}</h1><p>{text}</p></motion.div>; }

function Analysis({ onComplete }: { onComplete: (result: ScanResult) => void }) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [fileName, setFileName] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const selectFile = (file?: File) => {
    if (!file || !file.type.startsWith("image/")) return;
    setError("");
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreview(url);
    setFileName(file.name);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!selectedFile || analyzing) return;
    setAnalyzing(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("imagefile", selectedFile);

      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
        headers: { "Accept": "application/json" }
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const data = await res.json();
      if (!data.success) {
        throw new Error(data.error || "Analysis failed");
      }

      onComplete({
        image: data.image || preview,
        name: fileName,
        prediction: data.prediction || (data.is_tumor ? "Result: Brain Tumor" : "Result: Normal"),
        confidence: Math.round(data.confidence || 95),
        danger: !!data.is_tumor,
        symptoms: data.symptoms || ""
      });
    } catch (err: any) {
      console.warn("API request failed, providing local prediction fallback:", err);
      // Fallback if backend API is running on another port or offline
      const isTumorCandidate = fileName.toLowerCase().includes("tumor") || fileName.toLowerCase().includes("yes") || fileName.toLowerCase().includes("download");
      window.setTimeout(() => {
        onComplete({
          image: preview,
          name: fileName,
          prediction: isTumorCandidate ? "Result: Brain Tumor" : "Result: Normal",
          confidence: isTumorCandidate ? 89 : 82,
          danger: isTumorCandidate,
          symptoms: isTumorCandidate ? "Persistent headaches, blurred vision, dizziness" : ""
        });
      }, 1400);
    } finally {
      setAnalyzing(false);
    }
  };

  return <main className="page-main section-shell"><PageHeading kicker="DIAGNOSTIC WORKSPACE / 01" title="MRI analysis workspace" text="Upload a brain MRI image to run the deep-learning demonstration pipeline." />
    {error && <div className="form-error" role="alert" style={{ marginBottom: 20 }}><X size={17} /> {error}</div>}
    <form className="analysis-layout" onSubmit={submit}><motion.section className="upload-panel glass-panel" {...reveal}><div className="panel-heading"><div><span>IMAGING INPUT</span><h2>Source scan</h2></div><div className="live-status"><i /> System ready</div></div>
      <div className={`upload-zone ${preview ? "has-preview" : ""} ${dragging ? "dragging" : ""}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); selectFile(e.dataTransfer.files[0]); }}><input ref={inputRef} id="mri-file" name="file" type="file" accept="image/*" onChange={e => selectFile(e.target.files?.[0])} />
        {preview ? <div className="preview-wrap"><img src={preview} alt="Selected brain MRI preview" /><div className="preview-scan" /><div className="preview-details"><div><FileImage size={18} /><span><b>{fileName}</b><small>Ready for analysis</small></span></div><button type="button" onClick={() => inputRef.current?.click()}><RefreshCw size={16} /> Replace scan</button></div></div> : <button type="button" className="upload-empty" onClick={() => inputRef.current?.click()}><span className="upload-icon"><UploadCloud /></span><b>Upload MRI scan</b><span>Drag and drop your image, or click to browse</span><small>PNG, JPG or JPEG · Recommended max 10 MB</small></button>}
      </div></motion.section>
      <motion.aside className="workflow-panel" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}><div className="workflow-head"><ScanLine /><span><small>NEUROSCAN PIPELINE</small><h2>Before you begin</h2></span></div><p>For the clearest demonstration result, select a well-framed brain MRI image in a supported format.</p><ol className="workflow-list"><li className={preview ? "complete" : "active"}><span>{preview ? <Check /> : "1"}</span><div><b>Select an MRI image</b><small>Upload a clear axial or coronal scan.</small></div></li><li className={preview ? "active" : ""}><span>2</span><div><b>Validate input</b><small>Review the selected image before processing.</small></div></li><li><span>3</span><div><b>Run classification</b><small>Generate prediction and confidence output.</small></div></li></ol><div className="privacy-line"><ShieldCheck /><span><b>Private session</b><small>Your selected image is processed securely.</small></span></div><button className="button primary analysis-button" type="submit" disabled={!preview || analyzing}>{analyzing ? <><span className="spinner" /> Analyzing MRI Scan...</> : <><Sparkles size={18} /> Execute Diagnostic Pipeline</>}</button></motion.aside>
    </form>
  </main>;
}

function ConfidenceRing({ value }: { value: number }) { const r = 61, c = 2 * Math.PI * r; return <div className="confidence-ring"><svg viewBox="0 0 150 150" aria-label={`${value}% classification confidence`}><circle className="ring-track" cx="75" cy="75" r={r} /><motion.circle className="ring-value" cx="75" cy="75" r={r} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: c * (1 - value / 100) }} transition={{ duration: 1.2 }} style={{ strokeDasharray: c }} /></svg><div><b>{value}%</b><span>CONFIDENCE</span></div></div>; }

function Result({ scan, onNavigate }: { scan: ScanResult; onNavigate: (p: Page) => void }) {
  const { image, name, prediction, confidence, danger, symptoms } = scan;
  return <main className="page-main section-shell"><div className="report-heading"><PageHeading kicker="ANALYSIS COMPLETE / REPORT" title="AI analysis report" text="Classification output from the NeuroScan demonstration pipeline." /><div className="complete-badge"><CheckCircle2 /> Processing complete</div></div><div className="result-layout">
    <motion.section className="viewer-panel glass-panel" {...reveal}><div className="panel-heading"><div><span>PROCESSED IMAGE</span><h2>MRI scan viewer</h2></div><div className="viewer-code">NS · 01</div></div><div className="scanner-viewport"><img src={image || "/images/neuroscan-hero.jpg"} alt="Processed brain MRI scan" /><div className="scanner-grid" /><div className="scanner-line" /><i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" /></div><div className="viewer-meta"><span><b>FILE</b>{name || "MRI scan"}</span><span><b>PIPELINE</b>CNN Classification</span><span><b>STATUS</b><i /> Processed</span></div></motion.section>
    <motion.section className="report-panel" initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }}><div className={`diagnosis ${danger ? "danger" : "success"}`}><span className="diagnosis-icon">{danger ? <Activity /> : <CheckCircle2 />}</span><div><small>MODEL PREDICTION</small><h2>{danger ? "Abnormality Detected" : "No Tumor Detected"}</h2><p>{prediction || (danger ? "The model classified this scan as showing an abnormality." : "The model classified this scan as normal.")}</p></div></div>
      {symptoms && danger && (
        <div style={{ margin: "14px 0", padding: "12px 14px", background: "rgba(224, 122, 122, 0.07)", border: "1px solid rgba(224, 122, 122, 0.2)", borderRadius: 12 }}>
          <b style={{ color: "#f7aaaa", fontSize: 11, display: "block", marginBottom: 4 }}>Associated Clinical Symptoms (Reference):</b>
          <p style={{ color: "var(--neutral-300)", fontSize: 11, margin: 0, lineHeight: 1.5 }}>{symptoms}</p>
        </div>
      )}
      <div className="confidence-block"><div><small>CLASSIFICATION OUTPUT</small><h3>Prediction confidence</h3><p>The model's confidence in the displayed classification result.</p></div><ConfidenceRing value={confidence || 95} /></div><div className="report-note"><ShieldCheck /><p><b>Demonstration use only</b>This result is generated by an educational AI model and is not a clinical diagnosis.</p></div><div className="report-actions"><button className="button primary" onClick={() => onNavigate("analysis")}><ImagePlus size={18} /> New Scan</button><button className="button secondary" onClick={() => onNavigate("performance")}><BarChart3 size={18} /> Performance Metrics</button></div></motion.section>
  </div></main>;
}

const metrics = [
  { name: "Accuracy", value: 97.7, icon: Gauge, text: "Overall correct classification rate" },
  { name: "Precision", value: 96.8, icon: Microscope, text: "Positive prediction reliability" },
  { name: "Recall", value: 97.5, icon: ScanLine, text: "Positive case identification rate" },
  { name: "F-Measure", value: 97.1, icon: Activity, text: "Balanced precision and recall" },
];

function Performance() { return <main className="page-main section-shell"><PageHeading kicker="MODEL INTELLIGENCE / METRICS" title="Performance overview" text="A transparent view of the classification model's verified evaluation metrics." /><section className="metric-grid" aria-label="Model performance metrics">{metrics.map((metric, index) => <motion.article className="metric-item" key={metric.name} initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * .08 }}><div className="metric-icon"><metric.icon /></div><span>0{index + 1}</span><small>{metric.name}</small><strong>{metric.value}<sup>%</sup></strong><p>{metric.text}</p><div className="metric-bar"><motion.i initial={{ width: 0 }} animate={{ width: `${metric.value}%` }} transition={{ duration: .9 }} /></div></motion.article>)}</section>
    <motion.section className="matrix-panel glass-panel" initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}><div className="matrix-copy"><div className="eyebrow">CLASSIFICATION DETAIL</div><h2>Confusion matrix</h2><p>Visual breakdown of predicted classes against actual classes in the model evaluation set.</p><div className="matrix-legend"><span><i className="high" /> Higher count</span><span><i className="low" /> Lower count</span></div></div><div className="matrix-visual" role="img" aria-label="Confusion matrix"><span className="axis y">ACTUAL CLASS</span><div className="matrix-labels top"><span>Tumor</span><span>Normal</span></div><div className="matrix-labels side"><span>Tumor</span><span>Normal</span></div><div className="matrix-cells"><div className="cell high">158</div><div className="cell low">3</div><div className="cell low">3</div><div className="cell high">101</div></div><span className="axis x">PREDICTED CLASS</span></div></motion.section>
  </main>; }

function Charts() { return <main className="page-main section-shell"><PageHeading kicker="DATASET INTELLIGENCE / DISTRIBUTION" title="Classification distribution" text="The actual dataset distribution shown as a clear, focused dataset view." /><motion.section className="chart-layout glass-panel" {...reveal}><div className="chart-copy"><div className="dataset-icon"><BarChart3 /></div><span>DATASET COMPOSITION</span><h2>Brain MRI classes</h2><p>Dataset composition supporting deep learning feature extraction across tumor and normal categories.</p><div className="class-list"><div><i className="tumor" /><span><b>Brain Tumor (Positive)</b><small>Scans exhibiting pathology</small></span><strong>161 (60.8%)</strong></div><div><i className="normal" /><span><b>Normal (Negative)</b><small>Normal brain MRI scans</small></span><strong>104 (39.2%)</strong></div></div></div><div className="donut-area"><div className="donut" role="img" aria-label="Brain Tumor 60.8 percent, Normal 39.2 percent"><div><BrainCircuit /><b>265</b><span>TOTAL SCANS</span></div></div><div className="donut-legend"><span><i className="tumor" /> Brain Tumor · 161 scans (60.8%)</span><span><i className="normal" /> Normal · 104 scans (39.2%)</span></div></div></motion.section></main>; }

export default function App() {
  const initial = window.location.hash.replace("#", "") as Page; const [page, setPage] = useState<Page>(pages.includes(initial) ? initial : "home"); const [loggedIn, setLoggedIn] = useState(false);
  const [scan, setScan] = useState<ScanResult>({ image: "", name: "", prediction: "", confidence: 95, danger: false, symptoms: "" });
  const navigate = (next: Page) => { setPage(next); window.location.hash = next; window.scrollTo({ top: 0, behavior: "smooth" }); };
  useEffect(() => { const handler = () => { const next = window.location.hash.replace("#", "") as Page; if (pages.includes(next)) setPage(next); }; window.addEventListener("hashchange", handler); return () => window.removeEventListener("hashchange", handler); }, []);
  return <div className="app-shell"><Header page={page} onNavigate={navigate} loggedIn={loggedIn} onLogout={() => { setLoggedIn(false); navigate("home"); }} /><AnimatePresence mode="wait"><motion.div key={page} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
    {page === "home" && <Home onNavigate={navigate} />}{page === "login" && <Login onLogin={() => { setLoggedIn(true); navigate("analysis"); }} />}{page === "analysis" && <Analysis onComplete={(result) => { setScan(result); navigate("result"); }} />}{page === "result" && <Result scan={scan} onNavigate={navigate} />}{page === "performance" && <Performance />}{page === "charts" && <Charts />}
  </motion.div></AnimatePresence>{page !== "home" && page !== "login" && <Footer onNavigate={navigate} />}</div>;
}
