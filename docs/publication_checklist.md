# What remains to be done if the library goes into open access

> In Russian: [`publication_checklist-rus.md`](publication_checklist-rus.md)

> **Status: publication is postponed.** The library is closed and is currently
> being run in on the author's projects. This file is frozen until a decision on
> publication — nothing listed here needs doing while no decision has been taken.
>
> What has already been done and remains in force: the rights to the code are
> personal, the licence is chosen (BSD 3-Clause), the bibliographic data of the
> articles are filled in. The rest is a plan for the future.

Nothing remains on the list that the author does not have: the single unfilled
field is the release DOI, and Zenodo issues that after the repository is
published.

## Rights: settled

The rights to the code belong to the author personally: the code was written
outside working hours and does not enter the organization's reporting. No
agreement with the organization is required; the author chooses the licence
alone.

**BSD 3-Clause** (`LICENSE`) was chosen — the licence of the numpy/scipy/pandas
stack. Clause 3 forbids using the author's name to endorse derived products.

## To fill in before release

### 1. The name in the copyright — done

`LICENSE` reads `Copyright (c) 2026, Anna Kruchinina`.

### 2. The `CITATION.cff` fields — done

The bibliographic data are filled in for three articles; no `ЗАПОЛНИТЬ` remains
in the file:

| Place | Article |
|---|---|
| `preferred-citation` | Kruchinina, Xu, 2026, RJB 30(1):108–114, DOI 10.15593/RJBiomech/2026.1.10 — Table 4 supplies the thresholds in `models.py` |
| `references[0]` | Kruchinina, Yakushev, 2018, MUMB 73(4):97–100, DOI 10.3103/S0027133018340052 — trajectory model |
| `references[1]` | Kruchinina, 2020, RJB 24(1):33–39, DOI 10.15593/RJBiomech/2020.1.04 — trajectory model |

The same bibliographic data are duplicated in `README.md` (the "How to cite"
section) and its Russian copy `README-rus.md`, in the header of
`saccade_model.py`, and in the comment on the table in `models.py` — when
editing, update every place.

### 3. The library DOI via Zenodo

The order is strict: Zenodo sees only public repositories, and it does not pick
up past releases.

1. make the repository `ankorm/eyetracking_analytics` public;
2. on zenodo.org, sign in via GitHub and, **before creating the release**, turn
   on the switch for this repository in the Repositories section;
3. create a release tagged `v0.2.0` — Zenodo will take the archive and issue a
   DOI.

There will be two DOIs. The **concept DOI** points at all versions at once — it
is the one that goes into `README.md` and into the `doi` field of
`CITATION.cff`. The version DOI is bound to a specific release and is not needed
in a general reference.

This is the single unfilled field in `CITATION.cff`.

## To check before opening the repository

* [x] there are no participants' data in the repository: the fixtures are
      synthetic, with a fixed seed (`tests/synthetic.py`);
* [x] there are no notebooks on the `public` branch; the substantive text has
      been moved into `docs/history/`;
* [ ] review the branch history for accidentally committed paths and addresses
      from the working environment.

## Open methodological questions

While the library is closed, incompatible API changes are permitted and there is
no hurry to settle them. But settle them we must **before** publication: after
it, a change of behaviour becomes a breaking change for someone else's code.

* the measure of spatial spread: standard deviation, the 95th percentile or the
  maximum (`docs/fixation_spread_criterion.md`);
* the justification of the split and merge constants
  (`docs/split_and_merge_constants.md`);
* whether `axis_order` changes only the order of the axes or the chirality as
  well (item C-11).
