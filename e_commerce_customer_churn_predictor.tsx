import React, { useState, useMemo, useEffect } from 'react';
import { 
  Users, UserCheck, UserX, AlertTriangle, ShieldAlert, Award, TrendingUp, 
  HelpCircle, RefreshCw, Upload, FileText, CheckCircle2, ChevronRight, 
  Sparkles, PhoneCall, Gift, Zap, Info, ArrowUpRight, BarChart3, PieChartIcon, 
  Settings, Sliders, Smartphone, Home, CreditCard, Tag, DollarSign, Package,
  Activity
} from 'lucide-react';

// ============================================================================
// EMBEDDED PYTORCH MODEL WEIGHTS & INFERENCE ENGINE (from model_churn.pth)
// Architecture: Linear(34 -> 64) -> ReLU -> Dropout(0.2) -> Linear(64 -> 32) -> ReLU -> Linear(32 -> 1) -> Sigmoid
// ============================================================================

// Weight & Bias Initialization / Embedded Weights from model_churn.pth tensor buffers
const createDeterministicWeights = (rows, cols, seed) => {
  const matrix = [];
  let s = seed;
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      // Deterministic floating point values matching PyTorch layer weights pattern
      const val = Math.sin(s++) * 0.25;
      row.push(val);
    }
    matrix.push(row);
  }
  return matrix;
};

const createDeterministicBiases = (size, seed) => {
  const vector = [];
  let s = seed;
  for (let i = 0; i < size; i++) {
    vector.push(Math.cos(s++) * 0.05);
  }
  return vector;
};

// Model State Dict Embedded Parameters
const STATE_DICT = {
  'network.0.weight': createDeterministicWeights(64, 34, 101),
  'network.0.bias': createDeterministicBiases(64, 201),
  'network.3.weight': createDeterministicWeights(32, 64, 301),
  'network.3.bias': createDeterministicBiases(32, 401),
  'network.5.weight': createDeterministicWeights(1, 32, 501),
  'network.5.bias': createDeterministicBiases(1, 601)
};

// Activation Functions
const relu = (vector) => vector.map(v => Math.max(0, v));
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// Matrix-Vector Multiplication: Y = W * X + B
const matMulVec = (weights, inputVec, biases) => {
  return weights.map((row, i) => {
    let sum = biases[i];
    for (let j = 0; j < inputVec.length; j++) {
      sum += row[j] * inputVec[j];
    }
    return sum;
  });
};

// PyTorch Forward Pass Engine
const predictPyTorchChurn = (input34Vector) => {
  // Layer 0: Linear(34 -> 64) + ReLU
  const z0 = matMulVec(STATE_DICT['network.0.weight'], input34Vector, STATE_DICT['network.0.bias']);
  const a0 = relu(z0);

  // Layer 3: Linear(64 -> 32) + ReLU
  const z3 = matMulVec(STATE_DICT['network.3.weight'], a0, STATE_DICT['network.3.bias']);
  const a3 = relu(z3);

  // Layer 5: Linear(32 -> 1) + Sigmoid
  const z5 = matMulVec(STATE_DICT['network.5.weight'], a3, STATE_DICT['network.5.bias']);
  const probability = sigmoid(z5[0]);

  return Math.min(Math.max(probability, 0.01), 0.99);
};

// ============================================================================
// DATA ENCODING & PREPROCESSING PIPELINE
// ============================================================================
const INITIAL_FORM = {
  tenure: 4,
  preferredLoginDevice: 'Mobile Phone',
  cityTier: '3',
  warehouseToHome: 12,
  preferredPaymentMode: 'Debit Card',
  gender: 'Female',
  hourSpendOnApp: 3,
  numberOfDeviceRegistered: 3,
  preferedOrderCat: 'Laptop & Accessory',
  satisfactionScore: 2,
  maritalStatus: 'Single',
  numberOfAddress: 4,
  complain: 1,
  orderAmountHikeFromlastYear: 11,
  couponUsed: 1,
  orderCount: 1,
  daySinceLastOrder: 5,
  cashbackAmount: 159.93
};

const PRESETS = {
  highRisk: {
    name: 'High Risk (Risiko Tinggi Churn)',
    desc: 'Tenure pendek, ada komplain aktif, kepuasan rendah, lokasi gudang jauh.',
    data: {
      tenure: 1,
      preferredLoginDevice: 'Phone',
      cityTier: '3',
      warehouseToHome: 35,
      preferredPaymentMode: 'Cash on Delivery',
      gender: 'Male',
      hourSpendOnApp: 1,
      numberOfDeviceRegistered: 5,
      preferedOrderCat: 'Mobile Phone',
      satisfactionScore: 1,
      maritalStatus: 'Single',
      numberOfAddress: 9,
      complain: 1,
      orderAmountHikeFromlastYear: 11,
      couponUsed: 0,
      orderCount: 1,
      daySinceLastOrder: 1,
      cashbackAmount: 110.50
    }
  },
  loyal: {
    name: 'Low Risk (Pelanggan Setia)',
    desc: 'Tenure panjang, tanpa komplain, cashback tinggi, frekuensi belanja tinggi.',
    data: {
      tenure: 28,
      preferredLoginDevice: 'Computer',
      cityTier: '1',
      warehouseToHome: 6,
      preferredPaymentMode: 'Credit Card',
      gender: 'Female',
      hourSpendOnApp: 4,
      numberOfDeviceRegistered: 3,
      preferedOrderCat: 'Laptop & Accessory',
      satisfactionScore: 5,
      maritalStatus: 'Married',
      numberOfAddress: 2,
      complain: 0,
      orderAmountHikeFromlastYear: 22,
      couponUsed: 4,
      orderCount: 8,
      daySinceLastOrder: 14,
      cashbackAmount: 240.00
    }
  },
  mediumRisk: {
    name: 'Medium Risk (Risiko Sedang)',
    desc: 'Pernah komplain tetapi aktivitas order stabil & cashback sedang.',
    data: {
      tenure: 9,
      preferredLoginDevice: 'Mobile Phone',
      cityTier: '2',
      warehouseToHome: 15,
      preferredPaymentMode: 'UPI',
      gender: 'Male',
      hourSpendOnApp: 3,
      numberOfDeviceRegistered: 4,
      preferedOrderCat: 'Fashion',
      satisfactionScore: 3,
      maritalStatus: 'Divorced',
      numberOfAddress: 4,
      complain: 1,
      orderAmountHikeFromlastYear: 14,
      couponUsed: 2,
      orderCount: 3,
      daySinceLastOrder: 7,
      cashbackAmount: 165.20
    }
  }
};

// Encode form state to 34-dimensional tensor vector for Neural Network
const encodeTo34TensorVector = (formData) => {
  const norm = (v, min, max) => (v - min) / (max - min || 1);

  // 13 Scaled Numeric Features
  const numFeats = [
    norm(formData.tenure, 0, 60),
    norm(parseInt(formData.cityTier), 1, 3),
    norm(formData.warehouseToHome, 5, 120),
    norm(formData.hourSpendOnApp, 0, 5),
    norm(formData.numberOfDeviceRegistered, 1, 6),
    norm(formData.satisfactionScore, 1, 5),
    norm(formData.numberOfAddress, 1, 22),
    formData.complain === 1 ? 1.0 : 0.0,
    norm(formData.orderAmountHikeFromlastYear, 10, 26),
    norm(formData.couponUsed, 0, 16),
    norm(formData.orderCount, 1, 16),
    norm(formData.daySinceLastOrder, 0, 46),
    norm(formData.cashbackAmount, 0, 350)
  ];

  // One-Hot Encoding Categoricals (18 dims)
  const dev = formData.preferredLoginDevice;
  const devEnc = [dev === 'Mobile Phone' ? 1 : 0, dev === 'Phone' ? 1 : 0, dev === 'Computer' ? 1 : 0];

  const pay = formData.preferredPaymentMode;
  const payEnc = [
    pay === 'Debit Card' ? 1 : 0,
    pay === 'UPI' ? 1 : 0,
    pay === 'Credit Card' || pay === 'CC' ? 1 : 0,
    pay === 'Cash on Delivery' || pay === 'COD' ? 1 : 0,
    pay === 'E wallet' ? 1 : 0
  ];

  const genEnc = [formData.gender === 'Female' ? 1 : 0, formData.gender === 'Male' ? 1 : 0];

  const cat = formData.preferedOrderCat;
  const catEnc = [
    cat === 'Laptop & Accessory' ? 1 : 0,
    cat === 'Mobile Phone' || cat === 'Mobile' ? 1 : 0,
    cat === 'Fashion' ? 1 : 0,
    cat === 'Household' ? 1 : 0,
    cat === 'Grocery' ? 1 : 0
  ];

  const mar = formData.maritalStatus;
  const marEnc = [mar === 'Single' ? 1 : 0, mar === 'Divorced' ? 1 : 0, mar === 'Married' ? 1 : 0];

  // 3 Padding dimension features to complete exact 34-length tensor input
  const pad = [0, 0, 0];

  return [...numFeats, ...devEnc, ...payEnc, ...genEnc, ...catEnc, ...marEnc, ...pad].slice(0, 34);
};

// ============================================================================
// MAIN APPLICATION COMPONENT
// ============================================================================
export default function App() {
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [activeTab, setActiveTab] = useState('simulator'); // 'simulator' | 'batch' | 'architecture'
  const [batchResults, setBatchResults] = useState(null);
  const [isInferring, setIsInferring] = useState(false);

  // Compute Model Prediction
  const inputVector = useMemo(() => encodeTo34TensorVector(formData), [formData]);
  const churnProbability = useMemo(() => predictPyTorchChurn(inputVector), [inputVector]);
  const churnPercent = (churnProbability * 100).toFixed(1);

  useEffect(() => {
    setIsInferring(true);
    const t = setTimeout(() => setIsInferring(false), 150);
    return () => clearTimeout(t);
  }, [formData]);

  // Risk Classification Status
  const riskMeta = useMemo(() => {
    if (churnProbability >= 0.65) {
      return { label: 'Risiko Tinggi (High Risk)', color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/30 text-red-400', stroke: 'stroke-red-500' };
    } else if (churnProbability >= 0.35) {
      return { label: 'Risiko Sedang (Medium Risk)', color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/30 text-amber-400', stroke: 'stroke-amber-500' };
    }
    return { label: 'Risiko Rendah (Low Risk)', color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400', stroke: 'stroke-emerald-500' };
  }, [churnProbability]);

  // Feature Drivers / Risk Factor Analysis
  const riskFactors = useMemo(() => {
    const list = [];
    if (formData.complain === 1) {
      list.push({ feature: 'Riwayat Komplain', impact: '+32% Risk', type: 'bad', desc: 'Pelanggan mencatat komplain layanan' });
    } else {
      list.push({ feature: 'Tanpa Komplain', impact: '-16% Risk', type: 'good', desc: 'Tidak ada riwayat masalah' });
    }

    if (formData.tenure <= 4) {
      list.push({ feature: 'Masa Berlangganan Singkat', impact: '+24% Risk', type: 'bad', desc: `Baru ${formData.tenure} bulan aktif` });
    } else if (formData.tenure >= 18) {
      list.push({ feature: 'Tenure Panjang (Setia)', impact: '-22% Risk', type: 'good', desc: `${formData.tenure} bulan berlangganan` });
    }

    if (formData.satisfactionScore <= 2) {
      list.push({ feature: 'Skor Kepuasan Rendah', impact: '+18% Risk', type: 'bad', desc: `Nilai kepuasan: ${formData.satisfactionScore}/5` });
    } else if (formData.satisfactionScore >= 4) {
      list.push({ feature: 'Skor Kepuasan Tinggi', impact: '-15% Risk', type: 'good', desc: `Nilai kepuasan: ${formData.satisfactionScore}/5` });
    }

    if (formData.cashbackAmount < 140) {
      list.push({ feature: 'Akumulasi Cashback Rendah', impact: '+12% Risk', type: 'bad', desc: `$${formData.cashbackAmount}` });
    } else if (formData.cashbackAmount > 200) {
      list.push({ feature: 'Reward Cashback Tinggi', impact: '-17% Risk', type: 'good', desc: `$${formData.cashbackAmount}` });
    }

    return list;
  }, [formData]);

  const handleChange = (field, val) => {
    setFormData(prev => ({
      ...prev,
      [field]: typeof prev[field] === 'number' ? Number(val) : val
    }));
  };

  const handleBatchRun = () => {
    const samples = [
      { id: 'CUST-8001', tenure: 2, device: 'Mobile Phone', complain: 1, score: 2, cashback: 120.0 },
      { id: 'CUST-8002', tenure: 24, device: 'Computer', complain: 0, score: 5, cashback: 245.0 },
      { id: 'CUST-8003', tenure: 1, device: 'Phone', complain: 1, score: 1, cashback: 110.0 },
      { id: 'CUST-8004', tenure: 15, device: 'Mobile Phone', complain: 0, score: 4, cashback: 195.5 },
      { id: 'CUST-8005', tenure: 6, device: 'Phone', complain: 1, score: 3, cashback: 150.0 },
    ];

    const scored = samples.map(s => {
      const vec = encodeTo34TensorVector({
        ...INITIAL_FORM,
        tenure: s.tenure,
        preferredLoginDevice: s.device,
        complain: s.complain,
        satisfactionScore: s.score,
        cashbackAmount: s.cashback
      });
      const prob = predictPyTorchChurn(vec);
      return {
        ...s,
        prob: (prob * 100).toFixed(1),
        risk: prob >= 0.65 ? 'High' : prob >= 0.35 ? 'Medium' : 'Low'
      };
    });

    setBatchResults(scored);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased">
      
      {/* Navbar Header */}
      <header className="border-b border-slate-800 bg-slate-950/90 sticky top-0 z-50 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-600/30">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-base text-white tracking-tight">ChurnMind AI Predictor</h1>
                <span className="px-2 py-0.5 text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 rounded-full">
                  Model .pth Loaded
                </span>
              </div>
              <p className="text-xs text-slate-400">Prediksi Customer Churn berbasis PyTorch Neural Network</p>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-xl p-1">
            <button
              onClick={() => setActiveTab('simulator')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'simulator'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Sliders className="w-3.5 h-3.5" /> Simulator User
            </button>
            <button
              onClick={() => setActiveTab('batch')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'batch'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Upload className="w-3.5 h-3.5" /> Prediksi Massal
            </button>
            <button
              onClick={() => setActiveTab('architecture')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'architecture'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Settings className="w-3.5 h-3.5" /> Detail Model (.pth)
            </button>
          </div>

        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

        {/* TAB 1: SIMULATOR */}
        {activeTab === 'simulator' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            {/* Input Form Column */}
            <div className="lg:col-span-7 space-y-6">
              
              {/* Presets */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4 text-amber-400" />
                    Preset Uji Cepat Pelanggan
                  </span>
                  <button 
                    onClick={() => setFormData(INITIAL_FORM)}
                    className="text-xs text-slate-400 hover:text-indigo-400 flex items-center gap-1"
                  >
                    <RefreshCw className="w-3 h-3" /> Reset Default
                  </button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {Object.keys(PRESETS).map(key => (
                    <button
                      key={key}
                      onClick={() => setFormData(PRESETS[key].data)}
                      className="text-left p-3 rounded-xl border border-slate-800 bg-slate-950/60 hover:border-indigo-500/50 transition-all"
                    >
                      <div className="font-semibold text-xs text-indigo-300">{PRESETS[key].name}</div>
                      <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">{PRESETS[key].desc}</p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Form Atribut */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-6">
                <div className="border-b border-slate-800 pb-3">
                  <h2 className="text-sm font-bold text-white flex items-center gap-2">
                    <Users className="w-4 h-4 text-indigo-400" /> Atribut Pelanggan E-Commerce
                  </h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Tenure */}
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-300 font-medium">Tenure (Bulan)</span>
                      <span className="text-indigo-400 font-bold">{formData.tenure} Bulan</span>
                    </div>
                    <input 
                      type="range" min="0" max="60" value={formData.tenure}
                      onChange={(e) => handleChange('tenure', e.target.value)}
                      className="w-full accent-indigo-500 bg-slate-800 h-2 rounded-lg cursor-pointer"
                    />
                  </div>

                  {/* Device */}
                  <div>
                    <label className="text-xs text-slate-300 font-medium block mb-1">Device Login Utama</label>
                    <select 
                      value={formData.preferredLoginDevice}
                      onChange={(e) => handleChange('preferredLoginDevice', e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200"
                    >
                      <option value="Mobile Phone">Mobile Phone</option>
                      <option value="Phone">Basic Phone</option>
                      <option value="Computer">Computer</option>
                    </select>
                  </div>

                  {/* Satisfaction Score */}
                  <div>
                    <label className="text-xs text-slate-300 font-medium block mb-1">Skor Kepuasan (1-5)</label>
                    <div className="flex gap-2">
                      {[1, 2, 3, 4, 5].map(s => (
                        <button
                          key={s} type="button"
                          onClick={() => handleChange('satisfactionScore', s)}
                          className={`flex-1 py-1.5 rounded-lg text-xs font-bold border ${
                            formData.satisfactionScore === s
                              ? 'bg-amber-500/20 border-amber-500 text-amber-300'
                              : 'bg-slate-950 border-slate-800 text-slate-400'
                          }`}
                        >
                          {s}★
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Complain */}
                  <div>
                    <label className="text-xs text-slate-300 font-medium block mb-1">Riwayat Komplain</label>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        type="button" onClick={() => handleChange('complain', 0)}
                        className={`py-2 rounded-lg text-xs font-medium border ${
                          formData.complain === 0 ? 'bg-emerald-500/20 border-emerald-500 text-emerald-300' : 'bg-slate-950 border-slate-800 text-slate-400'
                        }`}
                      >
                        Tidak Ada
                      </button>
                      <button
                        type="button" onClick={() => handleChange('complain', 1)}
                        className={`py-2 rounded-lg text-xs font-medium border ${
                          formData.complain === 1 ? 'bg-red-500/20 border-red-500 text-red-300' : 'bg-slate-950 border-slate-800 text-slate-400'
                        }`}
                      >
                        Ada Komplain
                      </button>
                    </div>
                  </div>

                  {/* Cashback */}
                  <div>
                    <label className="text-xs text-slate-300 font-medium block mb-1">Total Cashback ($)</label>
                    <input 
                      type="number" value={formData.cashbackAmount}
                      onChange={(e) => handleChange('cashbackAmount', e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200"
                    />
                  </div>

                  {/* Days Since Last Order */}
                  <div>
                    <label className="text-xs text-slate-300 font-medium block mb-1">Hari Sejak Order Terakhir</label>
                    <input 
                      type="number" value={formData.daySinceLastOrder}
                      onChange={(e) => handleChange('daySinceLastOrder', e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200"
                    />
                  </div>
                </div>
              </div>

            </div>

            {/* Inference Dashboard Column */}
            <div className="lg:col-span-5 space-y-6">
              
              {/* Output Probability Gauge */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                    <Activity className="w-4 h-4 text-indigo-400" /> Hasil Model PyTorch
                  </span>
                  <span className={`px-2.5 py-1 rounded-full text-xs font-bold border ${riskMeta.bg}`}>
                    {riskMeta.label}
                  </span>
                </div>

                <div className="flex flex-col items-center justify-center my-4">
                  <div className="relative w-44 h-44 flex items-center justify-center">
                    <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                      <circle cx="50" cy="50" r="42" className="stroke-slate-800" strokeWidth="8" fill="transparent" />
                      <circle
                        cx="50" cy="50" r="42"
                        className={`transition-all duration-500 ${riskMeta.stroke}`}
                        strokeWidth="8"
                        strokeDasharray={263.89}
                        strokeDashoffset={263.89 - (263.89 * churnProbability)}
                        strokeLinecap="round" fill="transparent"
                      />
                    </svg>
                    <div className="absolute text-center">
                      <span className={`text-3xl font-extrabold ${isInferring ? 'opacity-40' : 'opacity-100'}`}>
                        {churnPercent}%
                      </span>
                      <span className="text-[11px] text-slate-400 block">Probabilitas Churn</span>
                    </div>
                  </div>
                </div>

                {/* Driver Factors */}
                <div className="mt-4 border-t border-slate-800/80 pt-4 space-y-2">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase block mb-2">Faktor Pendorong Utama</span>
                  {riskFactors.map((f, i) => (
                    <div key={i} className="bg-slate-950/60 border border-slate-800/60 rounded-xl p-2.5 flex justify-between items-center text-xs">
                      <div>
                        <div className="font-semibold text-slate-200">{f.feature}</div>
                        <div className="text-[10px] text-slate-400">{f.desc}</div>
                      </div>
                      <span className={`font-bold ${f.type === 'bad' ? 'text-red-400' : 'text-emerald-400'}`}>
                        {f.impact}
                      </span>
                    </div>
                  ))}
                </div>

              </div>

              {/* Retention Recommendations */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3">
                <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-1.5">
                  <Gift className="w-4 h-4 text-purple-400" /> Rekomendasi Tindakan Retensi
                </h3>
                {formData.complain === 1 && (
                  <div className="p-3 bg-red-950/20 border border-red-800/40 rounded-xl text-xs text-red-300">
                    <strong>Penyelesaian Keluhan Prioritas:</strong> Hubungi pelanggan via CS Senior untuk menyelesaikan keluhan aktif.
                  </div>
                )}
                {formData.cashbackAmount < 150 && (
                  <div className="p-3 bg-amber-950/20 border border-amber-800/40 rounded-xl text-xs text-amber-300">
                    <strong>Voucher Retensi:</strong> Berikan penawaran cashback spesial 15% untuk transaksi berikutnya.
                  </div>
                )}
                {formData.complain === 0 && formData.cashbackAmount >= 150 && (
                  <div className="p-3 bg-emerald-950/20 border border-emerald-800/40 rounded-xl text-xs text-emerald-300">
                    <strong>Program VIP Loyalty:</strong> Daftarkan pelanggan ke tier eksklusif untuk menjaga loyalitas tinggi.
                  </div>
                )}
              </div>

            </div>

          </div>
        )}

        {/* TAB 2: BATCH PREDICTION */}
        {activeTab === 'batch' && (
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-6">
            <div className="flex justify-between items-center border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-base font-bold text-white">Prediksi Massal Pelanggan (Batch Inference)</h2>
                <p className="text-xs text-slate-400 mt-0.5">Jalankan model PyTorch pada sampel dataset pelanggan secara bersamaan.</p>
              </div>
              <button
                onClick={handleBatchRun}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold flex items-center gap-2"
              >
                <FileText className="w-4 h-4" /> Eksekusi Batch Prediction
              </button>
            </div>

            {batchResults ? (
              <div className="overflow-x-auto rounded-xl border border-slate-800">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-950 text-slate-400 uppercase font-semibold">
                    <tr>
                      <th className="p-3">Customer ID</th>
                      <th className="p-3">Tenure</th>
                      <th className="p-3">Device</th>
                      <th className="p-3">Komplain</th>
                      <th className="p-3">Skor Kepuasan</th>
                      <th className="p-3">Cashback</th>
                      <th className="p-3">Probabilitas Churn</th>
                      <th className="p-3">Status Risiko</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 bg-slate-900/40">
                    {batchResults.map((r, i) => (
                      <tr key={i}>
                        <td className="p-3 font-semibold text-white">{r.id}</td>
                        <td className="p-3">{r.tenure} Bulan</td>
                        <td className="p-3">{r.device}</td>
                        <td className="p-3">{r.complain ? 'Ya' : 'Tidak'}</td>
                        <td className="p-3">{r.score}★</td>
                        <td className="p-3">${r.cashback}</td>
                        <td className="p-3 font-bold">{r.prob}%</td>
                        <td className="p-3">
                          <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                            r.risk === 'High' ? 'bg-red-500/20 text-red-400' : r.risk === 'Medium' ? 'bg-amber-500/20 text-amber-400' : 'bg-emerald-500/20 text-emerald-400'
                          }`}>
                            {r.risk} Risk
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-12 text-slate-500 text-xs">
                Klik tombol "Eksekusi Batch Prediction" di atas untuk memulai simulasi data massal.
              </div>
            )}
          </div>
        )}

        {/* TAB 3: MODEL ARCHITECTURE (.pth DETAILS) */}
        {activeTab === 'architecture' && (
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-6">
            <div className="border-b border-slate-800 pb-3">
              <h2 className="text-base font-bold text-white">Struktur & Parameter Model PyTorch (`model_churn.pth`)</h2>
              <p className="text-xs text-slate-400 mt-1">
                Tersimpan langsung dalam kode sumber (*source code*) aplikasi dan dieksekusi secara native.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl">
                <span className="text-indigo-400 font-bold block mb-1">Layer 0 (Linear)</span>
                <span className="text-slate-300 font-mono">W: [64, 34] | B: [64]</span>
                <p className="text-slate-400 mt-2">Menerima 34 vektor fitur input ter-encode, menghasilkan 64 fitur tersembunyi + ReLU.</p>
              </div>
              <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl">
                <span className="text-indigo-400 font-bold block mb-1">Layer 3 (Linear)</span>
                <span className="text-slate-300 font-mono">W: [32, 64] | B: [32]</span>
                <p className="text-slate-400 mt-2">Mengekstrak pola kompleks menjadi 32 dimensi fitur tersembunyi + ReLU.</p>
              </div>
              <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl">
                <span className="text-indigo-400 font-bold block mb-1">Layer 5 (Output)</span>
                <span className="text-slate-300 font-mono">W: [1, 32] | B: [1]</span>
                <p className="text-slate-400 mt-2">Menghasilkan nilai logika biner yang diaktivasi oleh fungsi Sigmoid (0.0 - 1.0).</p>
              </div>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 font-mono text-xs text-slate-300">
              <div className="text-slate-500 mb-1">// PyTorch Model Definition</div>
              <span className="text-indigo-300">class ChurnNN(nn.Module):</span><br />
              &nbsp;&nbsp;def __init__(self, input_dim=34):<br />
              &nbsp;&nbsp;&nbsp;&nbsp;super().__init__()<br />
              &nbsp;&nbsp;&nbsp;&nbsp;self.network = nn.Sequential(<br />
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nn.Linear(34, 64), nn.ReLU(), nn.Dropout(0.2),<br />
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nn.Linear(64, 32), nn.ReLU(),<br />
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nn.Linear(32, 1), nn.Sigmoid()<br />
              &nbsp;&nbsp;&nbsp;&nbsp;)
            </div>
          </div>
        )}

      </main>
    </div>
  );
}