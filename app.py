import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# -----------------------------
# Streamlit temel ayar
# -----------------------------
st.set_page_config(page_title="SA-6 Observer / State Estimation Lab", page_icon="👁️")

st.title("👁️ SA-6 – Observer / State Estimation Lab (Yay–Kütle Sistemi)")
st.write(
    """
Bu laboratuvarda **yay–kütle sisteminde** sadece konumu ölçebildiğimizi varsayıyoruz.
Gerçek hızı ölçemiyoruz; bunun yerine **Luenberger tipi bir gözlemci** ile:

- Konumu \\(\\hat{x}(t)\\)
- Hızı \\(\\hat{v}(t)\\)

tahmin etmeye çalışıyoruz.

Farklı gözlemci kazançları (L1, L2) ve ölçüm gürültü seviyeleri ile
tahminlerin nasıl düzeldiğini ve gürültüye nasıl tepki verdiğini inceleyebilirsin.
"""
)

st.markdown("---")


# -----------------------------
# Sistem parametreleri
# -----------------------------
st.subheader("1️⃣ Yay–Kütle Sistem Parametreleri")

col_sys1, col_sys2, col_sys3 = st.columns(3)
with col_sys1:
    m = st.slider(
        "Kütle m",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
    )
with col_sys2:
    k = st.slider(
        "Yay sabiti k",
        min_value=0.5,
        max_value=10.0,
        value=4.0,
        step=0.5,
        help="k büyüdükçe yay daha sert; salınım frekansı artar.",
    )
with col_sys3:
    c = st.slider(
        "Sönüm katsayısı c",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="c = 0: sönümsüz, c > 0: sürtünme ile sönümlü salınım.",
    )

st.write(f"Sistem parametreleri: **m = {m:.1f}**, **k = {k:.1f}**, **c = {c:.1f}**")


# -----------------------------
# Başlangıç koşulları (gerçek ve gözlemci)
# -----------------------------
st.subheader("2️⃣ Başlangıç Koşulları")

col_ic1, col_ic2 = st.columns(2)

with col_ic1:
    x0 = st.slider(
        "Gerçek başlangıç konumu x₀",
        min_value=-5.0,
        max_value=5.0,
        value=1.5,
        step=0.1,
    )
    v0 = st.slider(
        "Gerçek başlangıç hızı v₀",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
    )

with col_ic2:
    xhat0 = st.slider(
        "Gözlemcinin başlangıç konumu ẋ̂₀",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
    )
    vhat0 = st.slider(
        "Gözlemcinin başlangıç hızı ṽ̂₀",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
    )

st.write(
    f"Gerçek başlangıç: x₀ = {x0:.2f}, v₀ = {v0:.2f} | "
    f"Gözlemci başlangıcı: x̂₀ = {xhat0:.2f}, v̂₀ = {vhat0:.2f}"
)


# -----------------------------
# Gözlemci kazançları ve gürültü
# -----------------------------
st.subheader("3️⃣ Gözlemci Kazançları ve Ölçüm Gürültüsü")

col_L1, col_L2, col_noise = st.columns(3)

with col_L1:
    L1 = st.slider(
        "L1 (konum hatası kazancı)",
        min_value=0.0,
        max_value=30.0,
        value=8.0,
        step=0.5,
    )
with col_L2:
    L2 = st.slider(
        "L2 (hız hatası kazancı)",
        min_value=0.0,
        max_value=30.0,
        value=15.0,
        step=0.5,
    )
with col_noise:
    noise_level = st.slider(
        "Ölçüm gürültü seviyesi",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="0: gürültü yok, 1: oldukça gürültülü konum sensörü.",
    )

st.write(
    f"Gözlemci kazançları: **L1 = {L1:.1f}**, **L2 = {L2:.1f}**, "
    f"gürültü seviyesi: **{noise_level:.2f}**"
)

st.caption(
    "Not: L1, L2 küçükse tahmin yavaş toparlanır; çok büyükse gürültüye hassas olup salınım yapabilir."
)


# -----------------------------
# Simülasyon ayarları
# -----------------------------
st.subheader("4️⃣ Simülasyon Ayarları")

col_time1, col_time2 = st.columns(2)
with col_time1:
    t_max = st.slider(
        "Toplam süre (s)",
        min_value=2.0,
        max_value=20.0,
        value=10.0,
        step=1.0,
    )
with col_time2:
    dt = st.slider(
        "Zaman adımı Δt",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
    )

n_steps = int(t_max / dt) + 1
st.write(
    f"Simülasyon: **{t_max:.1f} s**, Δt = **{dt:.3f} s**, adım ≈ **{n_steps}**"
)


# -----------------------------
# Simülasyon fonksiyonu
# -----------------------------
def simulate_observer(m, k, c, x0, v0, xhat0, vhat0, L1, L2, dt, n_steps, noise_level):
    """
    Yay-kütle sisteminde sadece konum ölçülür.
    Gerçek sistem: X' = A X
    Observer: Xhat' = A Xhat + L (y - yhat)
    """
    # Sistem matrisi A
    A = np.array([[0.0, 1.0],
                  [-k / m, -c / m]])

    # Gözlemci kazanç vektörü
    L = np.array([[L1],
                  [L2]])

    t = np.zeros(n_steps)
    X = np.zeros((2, n_steps))      # [x; v]
    Xhat = np.zeros((2, n_steps))   # [x_hat; v_hat]
    y_meas = np.zeros(n_steps)

    # Başlangıçlar
    X[:, 0] = [x0, v0]
    Xhat[:, 0] = [xhat0, vhat0]

    rng = np.random.default_rng(0)

    for n in range(n_steps - 1):
        # Gerçek durum ve ölçüm (konum)
        x, v = X[:, n]
        y_true = x
        noise = noise_level * rng.standard_normal()
        y_meas[n] = y_true + noise

        # Observer'ın tahmin ettiği çıktı
        xhat, vhat = Xhat[:, n]
        yhat = xhat

        # Gerçek sistem dinamiği: X_{n+1} = X_n + A X_n dt
        dX = A @ X[:, n]
        X[:, n + 1] = X[:, n] + dX * dt

        # Observer dinamiği: Xhat' = A Xhat + L (y_meas - yhat)
        innovation = (y_meas[n] - yhat)
        dXhat = (A @ Xhat[:, n]) + (L[:, 0] * innovation)
        Xhat[:, n + 1] = Xhat[:, n] + dXhat * dt

        t[n + 1] = t[n] + dt

    # Son adımın ölçümü
    x_last = X[0, -1]
    y_meas[-1] = x_last + noise_level * rng.standard_normal()

    return t, X, Xhat, y_meas


# Simülasyonu çalıştır
t, X, Xhat, y_meas = simulate_observer(
    m, k, c, x0, v0, xhat0, vhat0, L1, L2, dt, n_steps, noise_level
)

x = X[0, :]
v = X[1, :]
xhat = Xhat[0, :]
vhat = Xhat[1, :]

e_x = x - xhat
e_v = v - vhat


# -----------------------------
# Konum: gerçek vs tahmin vs ölçüm
# -----------------------------
st.markdown("---")
st.subheader("5️⃣ Konum – Gerçek vs Tahmin vs Ölçüm")

fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(t, x, label="Gerçek konum x(t)")
ax1.plot(t, xhat, label="Tahmin edilen konum x̂(t)")
ax1.plot(t, y_meas, alpha=0.4, linestyle=":", label="Ölçülen (gürültülü) konum y_meas")
ax1.set_xlabel("t (s)")
ax1.set_ylabel("Konum")
ax1.set_title("Konum: Gerçek vs Observer vs Ölçüm")
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax1.legend()

st.pyplot(fig1)


# -----------------------------
# Hız: gerçek vs tahmin
# -----------------------------
st.subheader("Hız – Gerçek vs Tahmin")

fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(t, v, label="Gerçek hız v(t)")
ax2.plot(t, vhat, label="Tahmin edilen hız v̂(t)")
ax2.set_xlabel("t (s)")
ax2.set_ylabel("Hız")
ax2.set_title("Hız: Gerçek vs Observer")
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax2.legend()

st.pyplot(fig2)


# -----------------------------
# Hata grafikleri
# -----------------------------
st.subheader("6️⃣ Hata Eğrileri (x − x̂, v − v̂)")

fig3, ax3 = plt.subplots(figsize=(7, 3))
ax3.plot(t, e_x, label="Konum hatası e_x = x − x̂")
ax3.plot(t, e_v, label="Hız hatası e_v = v − v̂")
ax3.set_xlabel("t (s)")
ax3.set_ylabel("Hata")
ax3.set_title("Observer Hata Dinamikleri")
ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax3.legend()

st.pyplot(fig3)


# -----------------------------
# İlk adımlar tablosu
# -----------------------------
st.subheader("7️⃣ İlk Adımların Tablosu")

max_rows = min(20, n_steps)
df = pd.DataFrame(
    {
        "t (s)": t[:max_rows],
        "x": x[:max_rows],
        "x_hat": xhat[:max_rows],
        "v": v[:max_rows],
        "v_hat": vhat[:max_rows],
        "e_x": e_x[:max_rows],
        "e_v": e_v[:max_rows],
    }
)

st.dataframe(
    df.style.format(
        {
            "t (s)": "{:.3f}",
            "x": "{:.3f}",
            "x_hat": "{:.3f}",
            "v": "{:.3f}",
            "v_hat": "{:.3f}",
            "e_x": "{:.3f}",
            "e_v": "{:.3f}",
        }
    )
)


# -----------------------------
# Öğretmen kutusu
# -----------------------------
st.markdown("---")
st.info(
    "Bu lab, sadece konum ölçümüne sahip bir yay–kütle sisteminde, "
    "Luenberger tipi gözlemci kullanarak hızın (ve konumun) nasıl tahmin edilebileceğini "
    "sezgisel olarak gösterir."
)

with st.expander("👩‍🏫 Öğretmen Kutusu – Luenberger Sezgisi ve Sorular (SA-6)"):
    st.write(
        r"""
**Luenberger Observer Sezgisi:**

- Gerçek sistem: \\(X' = A X\\), çıktı: \\(y = C X\\) (burada \\(C = [1 \; 0]\\)).  
- Gözlemci:  

  \\[
  \hat{X}' = A \hat{X} + L (y - \hat{y}), \quad \hat{y} = C \hat{X}
  \\]

- \\(y - \hat{y}\\) ifadesi, **'ölçüm − tahmin'**, yani gözlemcinin hatasıdır.  
- L kazançları bu hatayı kullanarak \\(\hat{X}\\)'i düzeltir.

---

**Önerilen Etkinlikler:**

1. Gürültü **yokken** (noise_level = 0):

   - L1 = L2 = 0 iken ne oluyor? (Observer sadece tahmini dinamiğini takip ediyor.)  
   - L1 ve L2'yi arttırdıkça hata eğrilerinin (e_x, e_v) daha hızlı sıfıra
     yaklaştığını gözlemleyin.

2. Gürültü **varken** (örneğin noise_level = 0.5):

   - L1, L2 çok büyük seçilirse x̂ ve v̂ eğrileri ne kadar gürültülü hale geliyor?  
   - L1, L2 orta seviyede iken (örneğin L1=8, L2=15) hem hızlı düzeltme
     hem de makul gürültü seviyesini nasıl yakalayabilirsiniz?

3. Başlangıç hatası senaryosu:

   - Gerçek x₀ = 1.5, v₀ = 0 iken gözlemciyi x̂₀ = 0, v̂₀ = 0'dan başlatın.  
   - L1, L2 küçük ve büyük olduğunda, gözlemcinin ne kadar sürede gerçeğe
     yaklaştığını karşılaştırın.

4. Tartışma:

   - Gerçek endüstriyel sistemlerde neden hız sensörü yerine 'observer ile
     tahmin' kullanmak isteyebiliriz? (maliyet, gürültü, mekanik zorluk vb.)  
   - Bu gözlemciden çıkan \\(\hat{x}, \hat{v}\\) değerleri, **durum geri besleme**
     (state feedback) kullanan daha gelişmiş denetleyiciler için nasıl girdi olabilir?
"""
    )

st.caption(
    "SA-6: Bu modül, lise/üniversite başı seviyesinde state estimation (observer) "
    "kavramına görsel ve sezgisel bir giriş sunar."
)
