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

## 2. Should `sweep/v2` get a nodal head at all?

**Partly answered.** Contours exist now, from the solver rather than the model.
On the sample 11,229-node bracket a solve takes about 4 s end to end. Screening
the same STEP takes about 1 s end to end, almost all of it meshing; the model
call itself is about 60 ms. So the question is no longer "can we draw a contour"
but "do we need one at triage speed".

**The question that remains.** `PeakGNN` pools the whole graph into one scalar,
so the model still cannot produce a field. `render.py` refuses rather than
painting the pooled value across every node, because a uniform plot of one
number is indistinguishable from a prediction at a glance. A nodal head is worth
a training cycle only if contours are wanted across many candidates rather than
on the one already chosen to look at.

**Why it is not mine to settle.** It costs a training run and it weakens a claim.
The pooled head exists for a reason recorded in `model.py`: a maximum over noisy
per-node estimates is a worse estimator of the peak than pooling the whole graph,
and the peak is the scalar the schema carries. A nodal head trades some of that
accuracy for a field. Whether the picture is worth the number is your call.

**What would settle it.** Whether you would act on a contour or look at one. If
looking, `render_load_case` already draws the inputs and may be enough; if
acting, the field needs both a head and per-node conformal bands, which is
CW-adaptive split-CP rather than the scalar Mondrian bands in use now.

**What it would cost.** No new data. Every shard already carries `y_nodal`, so
this is a training run against data on disk, not another sweep.

**Until then.** Scalar only from the model, and the renderer refuses a
predicted field rather than inventing one. `surrogate_solve` gives the solver's
field in seconds.

---

## 3. Should displacement be a second head?

**Two questions that used to be one.** You asked early on for deformation
contours alongside von Mises. On the solver side that is no longer a question.
`generate/solver.py` always computed displacement, with E fixed at 1, so the
real deformation is that array divided by E in MPa. The only thing missing was
plumbing to keep it, which a solve path can add without touching the training
labels. What remains open is a deformation head on the **surrogate**.

**The question.** Should the model also predict displacement?

**Why it is not mine to settle.** A displacement head doubles the output.
More importantly, it has **no closed-form anchor in `mechanics`**: the
crosscheck validates stress, not displacement fields. A second output nothing
can check is a second thing to trust on faith, which is the opposite of how the
rest of this works. Displacement also depends on E, which the stress target was
deliberately made independent of.

**What would settle it.** Whether you need deformation at triage speed across
many candidates, or only on the part you have already chosen to solve. The
solver gives the second in seconds.

**What it would cost.** A new sweep. The shards carry `tau`, `peak` and the
nodal von Mises field, not displacement, so a head needs displacement labels
generated first. It is not a training run against data on disk.

**Sits behind question 2.** Displacement is a second field, and there is not
yet a first one from the model. Answering this before the nodal-head question
gets the order wrong.

**Until then.** The model screens stress only. Deformation comes from solving.

---

## 4. How much variation in load pattern?

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

## 5. When is the dataset big enough?

**The question.** 3000 samples is a guess. The honest answer is whatever makes
the conformal band stop narrowing.

**Why it is not mine to settle.** It trades your CPU time against band width,
and how tight the band must be depends on what decision rests on it.

**What would settle it.** Train at 1000, 2000 and the full set and plot the band
against sample count. A curve that has flattened says more data will not help.

**Until then.** 3000, which is an hour of CPU and enough to see whether the band
improves on SFEM's ×/÷ 19.75 at all.

---

## 6. Re-sweep and retrain the spring family?

**The question.** `addPipe` left its profile face in the model, so every swept
spring carried a duplicated end face — 24 nodes on the cap and 25 more on an
orphan 0.044 mm away, sharing none. Real CAD has no such thing, so the shipped
weights were fitted on a part that cannot arrive through the front door. Fed
sweep-path features the model scores 1.02 on springs; fed a STEP of the same
spring, 0.006.

**Why it is not mine to settle.** The geometry is already fixed, but clearing it
properly means regenerating 500 spring samples and retraining. That invalidates
`calibration.set_sha256`, so the dataset digest, the published Hugging Face
dataset and every band move together. It is a release, not a patch.

**What would settle it.** Whether springs are a family you need screened at all.
If they are not, deleting the family is cheaper and more honest than carrying a
sixth of the training set that nothing uses.

**Until then.** `spring` is in `predict.UNTRANSFERABLE_FAMILIES` and refused at
screening time. Reporting an estimate a hundred times below the solved value
would be much worse than reporting nothing. See
[DEPLOYED_COVERAGE.md](DEPLOYED_COVERAGE.md).

---

## 7. Is bracket's 0.82 worth chasing?

**The question.** After the selector repair, five families sit at or near 0.90.
Bracket came back at 0.83 and then 0.82 on two independent 100-part runs. The
standard error at n=100 is 0.030, so it is more than two standard errors low
both times. Its median pred/true of 0.86 says the band is centred slightly low,
not that it is too narrow. See
[DEPLOYED_COVERAGE.md](DEPLOYED_COVERAGE.md).

**Why it is not mine to settle.** The shortfall is now reproducible, so it is
real. Fixing it means more calibration data, or a closer look at how bracket
geometry is drawn, or a recalibration that shifts the bracket band. None of
those is worth doing unless the bracket band matters for a decision you are
actually making.

**What would settle it.** Whether bracket screening carries a decision. If it
does, start from the median pred/true. A band centred low is a bias to correct,
not a width to inflate.

**Until then.** Reported as measured, 0.82, and not rounded up to nominal.

---

## Answered

### Should the load-case image travel inside the document? — **no, it stays here**

`SurrogateScreening/v1` refuses unknown top-level keys, and a `fields[]` entry
requires a per-node interval because it describes a *predicted field*. A load
case is an input, not a prediction, so it fits neither. The options were to
widen the schema on both sides or to keep the image local.

Answered by the person who owns both: the two are independent agents. The
contract carries what the consumer enforces on arrival and nothing else, and
`render_load_case` writes the diagram next to the screened STEP for whoever is
looking at the part.

*Left open by this:* nothing in the contract. If a consumer ever needs the
diagram programmatically rather than a person needing it visually, this is worth
reopening -- but a visual check is exactly what it is for.


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
