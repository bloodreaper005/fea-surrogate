# Open questions

Decisions that belong to a person, not to a default. Each one records why it
cannot be settled from the code, what evidence would settle it, and what happens
in the meantime if nobody answers.

A question leaves this file when it is answered, and the answer goes into the
commit that acts on it.

---

## 1. Is 0.27 to 0.42 the right Poisson envelope?

**The question.** The sweep draws `nu` uniformly over 0.27–0.42, chosen to span
steel, aluminium and most engineering polymers. Is that the envelope your
designs actually live in?

**Why it is not mine to settle.** It sets what the domain gate admits. Too
narrow and it refuses work it could have screened; too wide and it accepts
queries the training data never covered. Only you know the materials.

**What would settle it.** The materials named in the product families this
would screen for.

**Until then.** 0.27–0.42, declared in the exported domain so a query outside
it is refused rather than silently extrapolated.

---

## 2. Should displacement be a second head?

**The question.** You asked early on for deformation contours alongside von
Mises. The solver computes displacement already; it simply is not a label.

**Why it is not mine to settle.** A displacement head doubles the output and,
more importantly, has **no closed-form anchor in `mechanics`** — the crosscheck
validates stress, not displacement fields. A second output nothing can check is
a second thing to trust on faith, which is the opposite of how the rest of this
works.

**What would settle it.** Whether a deformation plot is something you would act
on, or something you would look at. If the former, it needs an anchor first.

**Until then.** Stress only. The displacement is computed and discarded, so
adding it later costs a re-sweep and nothing else.

---

## 3. How much variation in load pattern?

**The question.** Each family applies one uniform traction over a fixed set of
faces. Real parts see distributed pressure, contact patches, and loads that move.

**Why it is not mine to settle.** It is the difference between a surrogate that
screens the load case you designed for and one that screens a family of load
cases. The second is much more useful and much more expensive, and the cost
falls on sweep time you have to wait for.

**What would settle it.** Whether the screening question is "is this part in
trouble under the load I specified" or "is this part in trouble at all".

**Until then.** One load case per sample, with location and direction in the
features — already the thing SFEM omitted.

---

## 4. When is the dataset big enough?

**The question.** 3000 samples is a guess. The honest answer is whatever makes
the conformal band stop narrowing.

**Why it is not mine to settle.** It trades your CPU time against band width,
and how tight the band must be depends on what decision rests on it.

**What would settle it.** Train at 1000, 2000 and the full set and plot the band
against sample count. A curve that has flattened says more data will not help.

**Until then.** 3000, which is an hour of CPU and enough to see whether the band
improves on SFEM's ×/÷ 19.75 at all.

---

## Answered

### One regression head, or a family embedding? — **one head, calibrated per family**

Settled by the first swept run, and the evidence pointed somewhere neither
option anticipated: the problem was the **band**, not the model.

A single head reached a shared band of ×/÷ 1.91 against SFEM's ×/÷ 19.75, and
marginal coverage of 0.904 — apparently dead on target. Per family:

| family | own band | coverage under the shared band |
| --- | --- | --- |
| pulley | ×/÷ 1.46 | 1.00 |
| bracket | ×/÷ 1.59 | 0.98 |
| stepped_shaft | ×/÷ 1.59 | 0.99 |
| flange | ×/÷ 1.66 | 0.98 |
| spring | ×/÷ 1.70 | 0.95 |
| **welded_t** | **×/÷ 2.99** | **0.58** |

The average was reached by over-covering five families and badly under-covering
one. A welded T joint was inside its stated interval 58% of the time against a
claimed 90%.

The model did not need conditioning — it predicts every family well enough that
five of six sit within ×/÷ 1.7. What needed conditioning was the calibration.
So: one head, which keeps the falsifiable claim that geometry alone determines
the answer and that a seventh family should work without retraining; and
Mondrian per-family conformal bands, which restore a per-family coverage
guarantee.

A part whose family is not named gets the **widest** band rather than the
average, because an estimate whose group is unknown has not been shown to be
covered at the stated rate by anything narrower.

*Left open by this:* whether the family must be declared at inference, or
whether residual-normalised conformal — scaling the band by predicted
difficulty — can recover per-group coverage with no label. The second is more
elegant and untested here.
