(function () {
    "use strict";

    // Bucket order for degree histograms; must match GraphStructureAnalyzer output.
    const BUCKET_ORDER = ["0-1", "2-10", "11-100", "101-1K", "1K-10K", "10K+"];

    function parseJSONTag(id) {
        const node = document.getElementById(id);
        if (!node) return {};
        const raw = (node.textContent || "").trim();
        if (!raw) return {};
        try {
            return JSON.parse(raw);
        } catch (e) {
            console.error("Failed to parse JSON tag #" + id, e);
            return {};
        }
    }

    function createElement(tag, attrs, ...children) {
        const el = document.createElement(tag);
        if (attrs) {
            for (const key of Object.keys(attrs)) {
                const val = attrs[key];
                if (val === null || val === undefined || val === false) continue;
                if (key === "className") el.className = val;
                else if (key === "text") el.textContent = val;
                else if (key === "hidden") el.hidden = Boolean(val);
                else el.setAttribute(key, val);
            }
        }
        for (const child of children) {
            if (child === null || child === undefined) continue;
            if (typeof child === "string" || typeof child === "number") {
                el.appendChild(document.createTextNode(String(child)));
            } else {
                el.appendChild(child);
            }
        }
        return el;
    }

    function formatNumber(n) {
        if (n === null || n === undefined) return "-";
        if (typeof n !== "number") return String(n);
        return n.toLocaleString("en-US");
    }

    function formatPercent(fraction) {
        if (fraction === null || fraction === undefined) return "-";
        return (fraction * 100).toFixed(2) + "%";
    }

    function classForThreshold(value, green, yellow) {
        // value <= green -> green, value <= yellow -> yellow, else red.
        if (value <= green) return "status-green";
        if (value <= yellow) return "status-yellow";
        return "status-red";
    }

    function classForNullRate(rate) {
        if (rate > 0.9) return "status-red";
        if (rate > 0.5) return "status-yellow";
        return "status-green";
    }

    function sumValues(obj) {
        if (!obj) return 0;
        let total = 0;
        for (const key of Object.keys(obj)) {
            const v = obj[key];
            if (typeof v === "number") total += v;
        }
        return total;
    }

    function hasAnyPositive(obj) {
        if (!obj) return false;
        for (const key of Object.keys(obj)) {
            if (obj[key] > 0) return true;
        }
        return false;
    }

    // ---- Rendering ----

    function renderHeader(analysis) {
        const metaEl = document.getElementById("report-meta");
        const cfgEl = document.getElementById("report-config-summary");
        const now = new Date().toISOString();
        metaEl.textContent = "Generated at " + now;

        const nodeTypes = Object.keys(analysis.node_counts || {});
        const edgeTypes = Object.keys(analysis.edge_counts || {});
        cfgEl.textContent =
            "Node tables: " + (nodeTypes.length ? nodeTypes.join(", ") : "(none)") +
            " | Edge tables: " + (edgeTypes.length ? edgeTypes.join(", ") : "(none)");
    }

    function overallStatus(analysis) {
        // Hard fails -> red.
        if (hasAnyPositive(analysis.duplicate_node_counts) ||
            hasAnyPositive(analysis.dangling_edge_counts) ||
            hasAnyPositive(analysis.referential_integrity_violations) ||
            hasAnyPositive(analysis.super_hub_int16_clamp_count)) {
            return "status-red";
        }
        // Check thresholded metrics for yellow.
        const totalNodes = sumValues(analysis.node_counts);
        if (totalNodes > 0) {
            const isolatedFrac = sumValues(analysis.isolated_node_counts) / totalNodes;
            const coldFrac = sumValues(analysis.cold_start_node_counts) / totalNodes;
            if (isolatedFrac > 0.05 || coldFrac > 0.10) return "status-red";
            if (isolatedFrac > 0.01 || coldFrac > 0.05) return "status-yellow";
        }
        // NULL rates.
        const nullRates = analysis.null_rates || {};
        for (const table of Object.keys(nullRates)) {
            for (const col of Object.keys(nullRates[table])) {
                const r = nullRates[table][col];
                if (r > 0.9) return "status-red";
            }
        }
        return "status-green";
    }

    function renderOverview(analysis) {
        const container = document.getElementById("overview-cards");
        const totalNodes = sumValues(analysis.node_counts);
        const totalEdges = sumValues(analysis.edge_counts);
        const nodeTypes = Object.keys(analysis.node_counts || {}).length;
        const edgeTypes = Object.keys(analysis.edge_counts || {}).length;
        const status = overallStatus(analysis);

        const cards = [
            ["Total nodes", formatNumber(totalNodes)],
            ["Total edges", formatNumber(totalEdges)],
            ["Node types", formatNumber(nodeTypes)],
            ["Edge types", formatNumber(edgeTypes)],
        ];
        for (const [label, value] of cards) {
            container.appendChild(createElement("div", { className: "card" },
                createElement("div", { className: "card-label", text: label }),
                createElement("div", { className: "card-value data-value", text: value })
            ));
        }
        const statusLabel = status === "status-green" ? "OK" :
                            status === "status-yellow" ? "WARNING" : "CRITICAL";
        container.appendChild(createElement("div", { className: "card" },
            createElement("div", { className: "card-label", text: "Overall status" }),
            createElement("div", { className: "card-value" },
                createElement("span", { className: status, text: statusLabel }))
        ));
    }

    function renderNullRates(analysis, queriesMap) {
        const container = document.getElementById("null-rates-container");
        const rates = analysis.null_rates || {};
        const rows = [];
        for (const table of Object.keys(rates)) {
            for (const col of Object.keys(rates[table])) {
                rows.push({ table: table, column: col, rate: rates[table][col] });
            }
        }
        if (rows.length === 0) {
            container.appendChild(createElement("p", { text: "No NULL rate data available." }));
            return;
        }
        const disc = renderQueryDisclosureByPrefix(
            queriesMap, "data_quality:null_rates:"
        );
        if (disc) container.appendChild(disc);
        rows.sort((a, b) => b.rate - a.rate);
        const thead = createElement("thead", null,
            createElement("tr", null,
                createElement("th", { text: "Table" }),
                createElement("th", { text: "Column" }),
                createElement("th", { text: "NULL rate" })));
        const tbody = createElement("tbody");
        for (const r of rows) {
            const cls = classForNullRate(r.rate);
            tbody.appendChild(createElement("tr", null,
                createElement("td", { text: r.table }),
                createElement("td", { text: r.column }),
                createElement("td", { className: "numeric" },
                    createElement("span", { className: cls, text: formatPercent(r.rate) }))));
        }
        container.appendChild(createElement("table", null, thead, tbody));
    }

    function renderIntegrity(analysis, queriesMap) {
        const container = document.getElementById("integrity-container");
        const integrityPrefixes = [
            "data_quality:duplicate_nodes:",
            "data_quality:duplicate_edges:",
            "data_quality:dangling_edges:",
            "data_quality:referential_integrity:",
            "graph_structure:self_loops:",
            "graph_structure:isolated_nodes:",
            "graph_structure:cold_start_nodes:",
        ];
        const aggregate = (queriesMap && Object.keys(queriesMap).length)
            ? createElement("details", { className: "query-disclosure" })
            : null;
        if (aggregate) {
            aggregate.appendChild(createElement("summary", { text: "Show SQL" }));
            let any = false;
            for (const prefix of integrityPrefixes) {
                for (const key of Object.keys(queriesMap)) {
                    if (key.indexOf(prefix) !== 0) continue;
                    for (const sql of (queriesMap[key] || [])) {
                        aggregate.appendChild(createElement("p", {
                            className: "sql-key",
                            text: key,
                        }));
                        aggregate.appendChild(createElement("pre", {
                            className: "sql",
                            text: sql,
                        }));
                        any = true;
                    }
                }
            }
            if (any) container.appendChild(aggregate);
        }
        const rows = [
            ["Duplicate nodes", analysis.duplicate_node_counts],
            ["Duplicate edges", analysis.duplicate_edge_counts],
            ["Dangling edges", analysis.dangling_edge_counts],
            ["Referential integrity violations", analysis.referential_integrity_violations],
            ["Self loops", analysis.self_loop_counts],
            ["Isolated nodes", analysis.isolated_node_counts],
            ["Cold-start nodes (degree 0-1)", analysis.cold_start_node_counts],
        ];
        const thead = createElement("thead", null,
            createElement("tr", null,
                createElement("th", { text: "Check" }),
                createElement("th", { text: "Per-type counts" }),
                createElement("th", { text: "Total" })));
        const tbody = createElement("tbody");
        for (const [label, obj] of rows) {
            const total = sumValues(obj);
            const isHardFail = (label === "Duplicate nodes" ||
                                label === "Dangling edges" ||
                                label === "Referential integrity violations");
            const cls = isHardFail
                ? (total > 0 ? "status-red" : "status-green")
                : (total > 0 ? "status-yellow" : "status-green");
            const detail = obj && Object.keys(obj).length
                ? Object.keys(obj).map(k => k + ": " + formatNumber(obj[k])).join(", ")
                : "(none)";
            tbody.appendChild(createElement("tr", null,
                createElement("td", { text: label }),
                createElement("td", { className: "data-value", text: detail }),
                createElement("td", { className: "numeric" },
                    createElement("span", { className: cls, text: formatNumber(total) }))));
        }
        container.appendChild(createElement("table", null, thead, tbody));
    }

    function relativeFacetsPath(resultKey, chunkIndex, totalChunks) {
        // result_key is "node:user" / "edge:engagement"; the FeatureProfiler
        // writes facets.html to {output_gcs_path}/feature_profiler/{kind}s/{type}/facets.html
        // for single-chunk tables, or to .../{type}/chunk_NN/facets.html when
        // the projection was split across multiple Dataflow pipelines.
        // Using a relative src means the embed and "full-screen" link both work
        // when the report folder is downloaded from GCS as-is.
        const parts = resultKey.split(":");
        if (parts.length !== 2) return null;
        const kind = parts[0];
        const typeName = parts[1];
        if (!kind || !typeName) return null;
        const total = typeof totalChunks === "number" ? totalChunks : 1;
        const idx = typeof chunkIndex === "number" ? chunkIndex : 0;
        const subdir = total > 1 ? "chunk_" + String(idx).padStart(2, "0") + "/" : "";
        return "feature_profiler/" + kind + "s/" + typeName + "/" + subdir + "facets.html";
    }

    function renderFeatureProfileErrors(container, errors) {
        if (!errors || !errors.length) return;
        const card = createElement("div", { className: "warning-box" });
        card.appendChild(createElement("strong", {
            text: "Feature profiling errors (" + errors.length + ")",
        }));
        const tbody = createElement("tbody");
        for (const err of errors) {
            const jobCell = createElement("td", { className: "data-value" });
            if (err.console_url) {
                const link = createElement("a", {
                    href: err.console_url,
                    target: "_blank",
                    rel: "noopener noreferrer",
                    text: err.job_name || err.job_id || "Open Dataflow job ↗",
                });
                jobCell.appendChild(link);
                if (err.job_id) {
                    jobCell.appendChild(createElement("br"));
                    jobCell.appendChild(createElement("span",
                        { className: "data-value", text: err.job_id }));
                }
            } else if (err.job_name || err.job_id) {
                jobCell.textContent = err.job_name || err.job_id;
            } else {
                jobCell.textContent = "—";
            }
            tbody.appendChild(createElement("tr", null,
                createElement("td", { text: err.result_key || "" }),
                createElement("td", { text: err.stage || "" }),
                createElement("td", { className: "data-value", text: err.bq_table || "" }),
                jobCell,
                createElement("td", { className: "data-value", text: err.message || "" })));
        }
        const table = createElement("table", null,
            createElement("thead", null, createElement("tr", null,
                createElement("th", { text: "Table key" }),
                createElement("th", { text: "Stage" }),
                createElement("th", { text: "BQ table" }),
                createElement("th", { text: "Dataflow job" }),
                createElement("th", { text: "Message" }))),
            tbody);
        const details = createElement("details", { open: "" },
            createElement("summary", { text: "Errors and skipped tables" }),
            table);
        container.appendChild(card);
        container.appendChild(details);
    }

    function renderFeatureStatistics(profile) {
        const section = document.getElementById("feature-statistics");
        const container = document.getElementById("feature-statistics-container");
        const facets = (profile && profile.facets_html_paths) || {};
        const errors = (profile && profile.errors) || [];
        const keys = Object.keys(facets);
        if (keys.length === 0 && errors.length === 0) {
            section.hidden = true;
            return;
        }
        section.hidden = false;
        renderFeatureProfileErrors(container, errors);
        for (const resultKey of keys) {
            // Sidecar may be either a list of GCS URIs (one per chunk for
            // wide tables that were split across multiple Dataflow pipelines)
            // or a bare string (legacy single-Facets shape). Normalize.
            const value = facets[resultKey];
            const paths = Array.isArray(value) ? value : (value ? [value] : []);
            const totalChunks = paths.length;
            const summary = createElement("summary", { text: "FACETS: " + resultKey });
            const details = createElement("details", { open: "" }, summary);

            for (let i = 0; i < paths.length; i++) {
                const relPath = relativeFacetsPath(resultKey, i, totalChunks);
                const absPath = paths[i] || "";
                if (totalChunks > 1) {
                    details.appendChild(createElement("div", {
                        className: "facets-chunk-caption",
                        text: "Chunk " + (i + 1) + " / " + totalChunks,
                    }));
                }
                if (relPath) {
                    details.appendChild(createElement("p", { className: "data-value" },
                        createElement("a", {
                            href: relPath,
                            target: "_blank",
                            rel: "noopener noreferrer",
                            text: "Open full-screen ↗",
                        }),
                        createElement("span", { text: "  (" + absPath + ")" }),
                    ));
                    details.appendChild(createElement("iframe", {
                        className: "facets-embed",
                        src: relPath,
                        sandbox: "allow-scripts allow-same-origin",
                    }));
                } else {
                    // Fall back to absolute path when the result_key is malformed.
                    details.appendChild(createElement("p", { className: "data-value" },
                        createElement("a", {
                            href: absPath,
                            target: "_blank",
                            rel: "noopener noreferrer",
                            text: absPath,
                        }),
                    ));
                }
            }
            container.appendChild(details);
        }
    }

    function renderEmbeddingDiagnostics(profile) {
        const section = document.getElementById("embedding-diagnostics");
        const container = document.getElementById("embedding-diagnostics-container");
        const diagnostics = (profile && profile.embedding_diagnostics) || {};
        const tableKeys = Object.keys(diagnostics);
        if (tableKeys.length === 0) {
            section.hidden = true;
            return;
        }
        section.hidden = false;
        const TOP_K_IN_REPORT = 5;
        for (const tableKey of tableKeys) {
            const perColumn = diagnostics[tableKey] || {};
            const columnKeys = Object.keys(perColumn);
            if (columnKeys.length === 0) continue;
            const tableNode = createElement("details", { open: "" },
                createElement("summary", { text: tableKey }));
            for (const colName of columnKeys) {
                const d = perColumn[colName] || {};
                const summary = createElement("p", { className: "embedding-summary" },
                    createElement("strong", { text: colName + ":" }),
                    " total=" + formatNumber(d.total),
                    ", unique=" + formatNumber(d.unique_count),
                    ", unique_ratio=" + formatPercent(d.unique_ratio));
                tableNode.appendChild(summary);

                const topK = Array.isArray(d.top_k) ? d.top_k.slice(0, TOP_K_IN_REPORT) : [];
                if (topK.length > 0) {
                    const thead = createElement("thead", null,
                        createElement("tr", null,
                            createElement("th", { text: "Hash" }),
                            createElement("th", { text: "Count" }),
                            createElement("th", { text: "Fraction" })));
                    const tbody = createElement("tbody");
                    for (const entry of topK) {
                        tbody.appendChild(createElement("tr", null,
                            createElement("td", { className: "numeric", text: String(entry.hash) }),
                            createElement("td", { className: "numeric data-value", text: formatNumber(entry.count) }),
                            createElement("td", { className: "numeric", text: formatPercent(entry.fraction) })));
                    }
                    tableNode.appendChild(createElement("table", null, thead, tbody));
                }
            }
            container.appendChild(tableNode);
        }
    }

    function renderCounts(analysis, queriesMap) {
        const container = document.getElementById("counts-container");
        const disc = renderQueryDisclosureByPrefix(
            queriesMap, "graph_structure:node_count:"
        );
        if (disc) container.appendChild(disc);
        const edgeDisc = renderQueryDisclosureByPrefix(
            queriesMap, "graph_structure:edge_count:"
        );
        if (edgeDisc) container.appendChild(edgeDisc);
        const thead = createElement("thead", null,
            createElement("tr", null,
                createElement("th", { text: "Type" }),
                createElement("th", { text: "Kind" }),
                createElement("th", { text: "Count" })));
        const tbody = createElement("tbody");
        for (const [name, count] of Object.entries(analysis.node_counts || {})) {
            tbody.appendChild(createElement("tr", null,
                createElement("td", { text: name }),
                createElement("td", { text: "node" }),
                createElement("td", { className: "numeric data-value", text: formatNumber(count) })));
        }
        for (const [name, count] of Object.entries(analysis.edge_counts || {})) {
            tbody.appendChild(createElement("tr", null,
                createElement("td", { text: name }),
                createElement("td", { text: "edge" }),
                createElement("td", { className: "numeric data-value", text: formatNumber(count) })));
        }
        container.appendChild(createElement("table", null, thead, tbody));
    }

    function renderDegreeHistogram(buckets, opts) {
        // Returns an SVG element for the given bucket counts.
        // opts (all optional):
        //   width, height: outer dimensions; default 720x220 for the
        //     full per-edge-type chart, override for sparkline mode.
        //   showLabels: when false, skips axis padding, value labels,
        //     bucket name labels and y-axis max — sparkline-style.
        //   sparkline: shorthand for sparkline styling (adds a CSS class
        //     so styles can override the regular histogram look).
        const o = opts || {};
        const width = o.width || 720;
        const height = o.height || 220;
        const showLabels = o.showLabels !== false;
        const padLeft = showLabels ? 50 : 2;
        const padRight = showLabels ? 10 : 2;
        const padTop = showLabels ? 16 : 2;
        const padBottom = showLabels ? 40 : 2;
        const innerW = width - padLeft - padRight;
        const innerH = height - padTop - padBottom;

        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("class", o.sparkline ? "histogram sparkline" : "histogram");
        svg.setAttribute("viewBox", "0 0 " + width + " " + height);

        const counts = BUCKET_ORDER.map(k => (buckets && buckets[k]) || 0);
        const maxCount = Math.max(1, ...counts);
        const barWidth = innerW / BUCKET_ORDER.length;
        const gap = showLabels ? 8 : 1;

        if (showLabels) {
            const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
            axis.setAttribute("class", "axis");
            axis.setAttribute("x1", padLeft);
            axis.setAttribute("y1", padTop + innerH);
            axis.setAttribute("x2", padLeft + innerW);
            axis.setAttribute("y2", padTop + innerH);
            svg.appendChild(axis);
        }

        for (let i = 0; i < BUCKET_ORDER.length; i++) {
            const c = counts[i];
            const h = (c / maxCount) * innerH;
            const x = padLeft + i * barWidth + gap / 2;
            const y = padTop + innerH - h;
            const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            rect.setAttribute("class", "bar");
            rect.setAttribute("x", x);
            rect.setAttribute("y", y);
            rect.setAttribute("width", Math.max(1, barWidth - gap));
            rect.setAttribute("height", h);
            svg.appendChild(rect);

            if (!showLabels) continue;

            const valueLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
            valueLabel.setAttribute("class", "value");
            valueLabel.setAttribute("x", x + (barWidth - gap) / 2);
            valueLabel.setAttribute("y", y - 4);
            valueLabel.setAttribute("text-anchor", "middle");
            valueLabel.textContent = formatNumber(c);
            svg.appendChild(valueLabel);

            const xLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
            xLabel.setAttribute("class", "label");
            xLabel.setAttribute("x", x + (barWidth - gap) / 2);
            xLabel.setAttribute("y", padTop + innerH + 16);
            xLabel.setAttribute("text-anchor", "middle");
            xLabel.textContent = BUCKET_ORDER[i];
            svg.appendChild(xLabel);
        }

        if (showLabels) {
            // Y-axis max label.
            const maxLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
            maxLabel.setAttribute("class", "label");
            maxLabel.setAttribute("x", padLeft - 6);
            maxLabel.setAttribute("y", padTop + 10);
            maxLabel.setAttribute("text-anchor", "end");
            maxLabel.textContent = formatNumber(maxCount);
            svg.appendChild(maxLabel);
        }

        return svg;
    }

    function renderQueryDisclosure(queriesMap, blockId) {
        // Returns a <details> with the SQL strings recorded under blockId,
        // or null when no queries were captured for that block.
        const queries = (queriesMap || {})[blockId] || [];
        if (!queries.length) return null;
        const det = createElement("details", { className: "query-disclosure" });
        det.appendChild(createElement("summary", { text: "Show SQL" }));
        for (const q of queries) {
            det.appendChild(createElement("pre", { className: "sql", text: q }));
        }
        return det;
    }

    function renderQueryDisclosureByPrefix(queriesMap, prefix) {
        // Aggregate disclosure — collects every block_id starting with
        // `prefix` into one expander. Used at section level when one
        // header summarizes data from many block_ids (e.g. NULL rates,
        // integrity counts).
        const matches = [];
        const map = queriesMap || {};
        for (const key of Object.keys(map)) {
            if (key.indexOf(prefix) !== 0) continue;
            const list = map[key] || [];
            for (const sql of list) matches.push({ key: key, sql: sql });
        }
        if (!matches.length) return null;
        const det = createElement("details", { className: "query-disclosure" });
        det.appendChild(createElement("summary", { text: "Show SQL" }));
        for (const entry of matches) {
            det.appendChild(createElement("p", {
                className: "sql-key",
                text: entry.key,
            }));
            det.appendChild(createElement("pre", {
                className: "sql",
                text: entry.sql,
            }));
        }
        return det;
    }

    function renderBlockHeader(level, title, queriesMap, blockId) {
        // <div class="block-header"><h{level}>title</h{level}>[<details>...]</div>
        // The disclosure is omitted when no queries are recorded for blockId.
        const wrap = createElement("div", { className: "block-header" });
        wrap.appendChild(createElement(level, { text: title }));
        const disc = renderQueryDisclosure(queriesMap, blockId);
        if (disc) wrap.appendChild(disc);
        return wrap;
    }

    function renderDegree(analysis, queriesMap) {
        const container = document.getElementById("degree-container");
        const degrees = analysis.degree_stats || {};
        const keys = Object.keys(degrees);
        if (keys.length === 0) {
            container.appendChild(createElement("p", { text: "No degree stats available." }));
            return;
        }
        for (const edgeType of keys) {
            const stats = degrees[edgeType];
            const median = stats.median || 1;
            const ratio = stats.p99 / Math.max(1, median);
            const ratioClass = classForThreshold(ratio, 50, 100);

            const statsLine = createElement("p", { className: "data-value" },
                "min=" + formatNumber(stats.min) +
                ", mean=" + (stats.mean !== undefined ? stats.mean.toFixed(2) : "-") +
                ", median=" + formatNumber(stats.median) +
                ", p90=" + formatNumber(stats.p90) +
                ", p99=" + formatNumber(stats.p99) +
                ", p99.9=" + formatNumber(stats.p999) +
                ", max=" + formatNumber(stats.max) +
                " | p99/median=",
                createElement("span", { className: ratioClass, text: ratio.toFixed(1) }));

            container.appendChild(renderBlockHeader(
                "h3", edgeType, queriesMap, "graph_structure:degree:" + edgeType
            ));
            container.appendChild(statsLine);
            container.appendChild(renderDegreeHistogram(stats.buckets || {}));
        }
    }

    function renderHubs(analysis, queriesMap) {
        const container = document.getElementById("hubs-container");
        const hubs = analysis.top_hubs || {};
        const keys = Object.keys(hubs);
        if (keys.length === 0) {
            container.appendChild(createElement("p", { text: "No hub data available." }));
            return;
        }
        for (const edgeType of keys) {
            container.appendChild(renderBlockHeader(
                "h3", edgeType, queriesMap, "graph_structure:top_hubs:" + edgeType
            ));
            const thead = createElement("thead", null,
                createElement("tr", null,
                    createElement("th", { text: "Rank" }),
                    createElement("th", { text: "Node ID" }),
                    createElement("th", { text: "Degree" })));
            const tbody = createElement("tbody");
            const rows = (hubs[edgeType] || []).slice(0, 20);
            rows.forEach((entry, i) => {
                const nodeId = Array.isArray(entry) ? entry[0] : entry.node_id;
                const degree = Array.isArray(entry) ? entry[1] : entry.degree;
                tbody.appendChild(createElement("tr", null,
                    createElement("td", { text: String(i + 1) }),
                    createElement("td", { className: "data-value", text: String(nodeId) }),
                    createElement("td", { className: "numeric data-value", text: formatNumber(degree) })));
            });
            container.appendChild(createElement("table", null, thead, tbody));
        }
    }

    function renderSuperHubWarning(analysis) {
        const box = document.getElementById("super-hub-warning");
        const clamps = analysis.super_hub_int16_clamp_count || {};
        const totalClamps = sumValues(clamps);
        if (totalClamps <= 0) {
            box.hidden = true;
            return;
        }
        box.hidden = false;
        box.className = "warning-box";
        const detail = Object.keys(clamps)
            .map(k => k + ": " + formatNumber(clamps[k]))
            .join(", ");
        box.appendChild(createElement("strong", { text: "Super-hub int16 clamp warning. " }));
        box.appendChild(document.createTextNode(
            formatNumber(totalClamps) + " node(s) exceed the int16 degree limit (32,767) and " +
            "will be silently clamped by GiGL. Per-type: " + detail
        ));
    }

    function renderSupervisionOverlap(analysis, queriesMap) {
        const section = document.getElementById("supervision-overlap");
        const container = document.getElementById("supervision-overlap-container");
        const stats = (analysis && analysis.supervision_cross_table_stats) || [];
        if (!stats.length) {
            section.hidden = true;
            return;
        }
        section.hidden = false;
        while (container.firstChild) container.removeChild(container.firstChild);

        for (const entry of stats) {
            const card = createElement("div", { className: "card" });
            const title = entry.driver_edge_type + " → " + entry.other_edge_type +
                " (" + entry.other_role + ")";
            // Card-level disclosure aggregating any block_id starting with
            // "supervision_overlap:<driver>:<other>:" — covers homogeneous
            // and heterogeneous anchor-column suffixes alike.
            const cardHeader = createElement("div", { className: "block-header" });
            cardHeader.appendChild(createElement("h3", { text: title }));
            const cardDisc = renderQueryDisclosureByPrefix(
                queriesMap,
                "supervision_overlap:" + entry.driver_edge_type +
                    ":" + entry.other_edge_type + ":"
            );
            if (cardDisc) cardHeader.appendChild(cardDisc);
            card.appendChild(cardHeader);
            card.appendChild(createElement("p", { className: "data-value" },
                "Anchor node type: ", entry.node_anchor,
                " (driver role: ", entry.driver_role, ")"));

            const driverPairs = entry.driver_pair_count || 0;
            const overlap = entry.overlap_pair_count || 0;
            const overlapFrac = driverPairs > 0 ? overlap / driverPairs : 0;
            const overlapClass = overlap === 0
                ? "status-green"
                : (overlapFrac >= 0.01 ? "status-red" : "status-yellow");

            const driverAnchors = entry.driver_anchor_count || 0;
            const zeroOther = entry.driver_anchors_with_zero_other || 0;
            const zeroFrac = driverAnchors > 0 ? zeroOther / driverAnchors : 0;
            const zeroClass = zeroFrac > 0.5
                ? "status-red"
                : (zeroFrac >= 0.05 ? "status-yellow" : "status-green");

            const driverName = entry.driver_edge_type;
            const otherName = entry.other_edge_type;
            const tbody = createElement("tbody");
            const rows = [
                [
                    "Distinct anchors in " + driverName,
                    formatNumber(driverAnchors),
                    null,
                ],
                [
                    "Distinct (anchor, neighbor) pairs in " + driverName,
                    formatNumber(driverPairs),
                    null,
                ],
                [
                    "Distinct (anchor, neighbor) pairs in " + otherName,
                    formatNumber(entry.other_pair_count || 0),
                    null,
                ],
                [
                    "Overlap pair count (" + driverName + " ∩ " + otherName + ")",
                    formatNumber(overlap) + "  (" + formatPercent(overlapFrac) + ")",
                    overlapClass,
                ],
                [
                    "Anchors in " + driverName + " with zero edges in " + otherName,
                    formatNumber(zeroOther) + "  (" + formatPercent(zeroFrac) + ")",
                    zeroClass,
                ],
                [
                    "Avg edges in " + otherName + " per anchor in " + driverName,
                    (entry.avg_other_per_driver_anchor || 0).toFixed(2),
                    null,
                ],
                [
                    "p50 / p90 / p99 / max edges in " + otherName +
                        " per anchor in " + driverName,
                    formatNumber(entry.p50_other_per_driver_anchor || 0) + " / " +
                    formatNumber(entry.p90_other_per_driver_anchor || 0) + " / " +
                    formatNumber(entry.p99_other_per_driver_anchor || 0) + " / " +
                    formatNumber(entry.max_other_per_driver_anchor || 0),
                    null,
                ],
            ];
            for (const [label, value, cls] of rows) {
                const valueCell = createElement("td", { className: "numeric data-value" });
                if (cls) {
                    valueCell.appendChild(createElement("span", { className: cls, text: value }));
                } else {
                    valueCell.textContent = value;
                }
                tbody.appendChild(createElement("tr", null,
                    createElement("td", { text: label }),
                    valueCell));
            }
            card.appendChild(createElement("table", null, tbody));
            container.appendChild(card);
        }
    }

    function renderNodeClassificationSupervision(analysis, queriesMap) {
        const section = document.getElementById("node-classification-supervision");
        const container = document.getElementById(
            "node-classification-supervision-container"
        );
        const stats = (analysis && analysis.node_classification_supervision_stats) || [];
        if (!stats.length) {
            section.hidden = true;
            return;
        }
        section.hidden = false;
        while (container.firstChild) container.removeChild(container.firstChild);

        for (const entry of stats) {
            const card = createElement("div", { className: "card" });
            card.appendChild(createElement(
                "h3",
                { text: "Node type: " + entry.node_type +
                       "   (label column: " + entry.label_column + ")" }
            ));
            const nt = entry.node_type;

            const sentinel = entry.sentinel_stats || {};
            const totalRows = sentinel.total_rows || 0;
            const nullCount = sentinel.null_count || 0;
            const validCount = sentinel.valid_label_count || 0;
            const validCoverage = sentinel.valid_label_coverage || 0;
            const sentinelCounts = sentinel.sentinel_counts || {};
            const sentinelTotal = Object.values(sentinelCounts)
                .reduce((acc, value) => acc + (value || 0), 0);

            const sentinelTbody = createElement("tbody");
            const sentinelRows = [
                ["Total rows", formatNumber(totalRows), null],
                [
                    "Valid labels (non-null AND non-sentinel)",
                    formatNumber(validCount) + "  (" + formatPercent(validCoverage) + ")",
                    validCoverage > 0 ? "status-green" : "status-red",
                ],
                [
                    "NULL labels",
                    formatNumber(nullCount),
                    nullCount > 0 ? "status-yellow" : null,
                ],
                [
                    "Sentinel labels (treated as missing)",
                    formatNumber(sentinelTotal),
                    sentinelTotal > 0 ? "status-yellow" : null,
                ],
            ];
            for (const [label, value, cls] of sentinelRows) {
                const valueCell = createElement("td", { className: "numeric data-value" });
                if (cls) {
                    valueCell.appendChild(createElement("span", { className: cls, text: value }));
                } else {
                    valueCell.textContent = value;
                }
                sentinelTbody.appendChild(createElement("tr", null,
                    createElement("td", { text: label }),
                    valueCell));
            }
            card.appendChild(renderBlockHeader(
                "h4", "Label hygiene", queriesMap,
                "nc_supervision:label_sentinel:" + nt
            ));
            card.appendChild(createElement("table", null, sentinelTbody));

            if (Object.keys(sentinelCounts).length) {
                const ulSentinel = createElement("ul");
                for (const [val, count] of Object.entries(sentinelCounts)) {
                    ulSentinel.appendChild(createElement("li", {
                        text: "sentinel " + JSON.stringify(val) + ": " +
                              formatNumber(count || 0),
                    }));
                }
                card.appendChild(ulSentinel);
            }

            const perClass = entry.per_class_degree || [];
            if (perClass.length) {
                card.appendChild(renderBlockHeader(
                    "h4", "Per-class degree", queriesMap,
                    "nc_supervision:per_class_degree:" + nt
                ));
                const tbody = createElement("tbody");
                tbody.appendChild(createElement("tr", null,
                    createElement("th", { text: "Class" }),
                    createElement("th", { text: "Count" }),
                    createElement("th", { text: "Cold-start (deg ≤ 1)" }),
                    createElement("th", { text: "Mean" }),
                    createElement("th", { text: "Median" }),
                    createElement("th", { text: "p90" }),
                    createElement("th", { text: "p99" }),
                    createElement("th", { text: "Max" }),
                    createElement("th", { text: "Distribution" })));
                for (const cls of perClass) {
                    const coldFrac = cls.count > 0
                        ? (cls.cold_start_count || 0) / cls.count
                        : 0;
                    const coldClass = coldFrac >= 0.5
                        ? "status-red"
                        : (coldFrac >= 0.1 ? "status-yellow" : "status-green");
                    const coldCell = createElement("td", { className: "numeric data-value" });
                    coldCell.appendChild(createElement("span", {
                        className: coldClass,
                        text: formatNumber(cls.cold_start_count || 0) +
                              "  (" + formatPercent(coldFrac) + ")",
                    }));
                    const distCell = createElement("td", { className: "sparkline-cell" });
                    distCell.appendChild(renderDegreeHistogram(cls.buckets || {}, {
                        width: 140,
                        height: 32,
                        showLabels: false,
                        sparkline: true,
                    }));
                    tbody.appendChild(createElement("tr", null,
                        createElement("td", { text: String(cls.class_value) }),
                        createElement("td", {
                            className: "numeric",
                            text: formatNumber(cls.count || 0),
                        }),
                        coldCell,
                        createElement("td", {
                            className: "numeric",
                            text: (cls.mean_degree || 0).toFixed(2),
                        }),
                        createElement("td", {
                            className: "numeric",
                            text: formatNumber(cls.median_degree || 0),
                        }),
                        createElement("td", {
                            className: "numeric",
                            text: formatNumber(cls.p90_degree || 0),
                        }),
                        createElement("td", {
                            className: "numeric",
                            text: formatNumber(cls.p99_degree || 0),
                        }),
                        createElement("td", {
                            className: "numeric",
                            text: formatNumber(cls.max_degree || 0),
                        }),
                        distCell));
                }
                card.appendChild(createElement("table", null, tbody));
            }

            const sentinelDegree = entry.sentinel_degree_stats || [];
            if (sentinelDegree.length) {
                card.appendChild(renderBlockHeader(
                    "h4", "Sentinel-label degree distribution", queriesMap,
                    "nc_supervision:per_class_degree:" + nt
                ));
                const tbody = createElement("tbody");
                tbody.appendChild(createElement("tr", null,
                    createElement("th", { text: "Sentinel" }),
                    createElement("th", { text: "Count" }),
                    createElement("th", { text: "Cold-start (deg ≤ 1)" }),
                    createElement("th", { text: "Mean" }),
                    createElement("th", { text: "Median" }),
                    createElement("th", { text: "p90" }),
                    createElement("th", { text: "p99" }),
                    createElement("th", { text: "Max" }),
                    createElement("th", { text: "Distribution" })));
                for (const cls of sentinelDegree) {
                    const coldFrac = cls.count > 0
                        ? (cls.cold_start_count || 0) / cls.count
                        : 0;
                    const coldClass = coldFrac >= 0.5
                        ? "status-red"
                        : (coldFrac >= 0.1 ? "status-yellow" : "status-green");
                    const coldCell = createElement("td", { className: "numeric data-value" });
                    coldCell.appendChild(createElement("span", {
                        className: coldClass,
                        text: formatNumber(cls.cold_start_count || 0) +
                              "  (" + formatPercent(coldFrac) + ")",
                    }));
                    const distCell = createElement("td", { className: "sparkline-cell" });
                    distCell.appendChild(renderDegreeHistogram(cls.buckets || {}, {
                        width: 140,
                        height: 32,
                        showLabels: false,
                        sparkline: true,
                    }));
                    tbody.appendChild(createElement("tr", null,
                        createElement("td", { text: String(cls.class_value) }),
                        createElement("td", {
                            className: "numeric",
                            text: formatNumber(cls.count || 0),
                        }),
                        coldCell,
                        createElement("td", {
                            className: "numeric",
                            text: (cls.mean_degree || 0).toFixed(2),
                        }),
                        createElement("td", {
                            className: "numeric",
                            text: formatNumber(cls.median_degree || 0),
                        }),
                        createElement("td", {
                            className: "numeric",
                            text: formatNumber(cls.p90_degree || 0),
                        }),
                        createElement("td", {
                            className: "numeric",
                            text: formatNumber(cls.p99_degree || 0),
                        }),
                        createElement("td", {
                            className: "numeric",
                            text: formatNumber(cls.max_degree || 0),
                        }),
                        distCell));
                }
                card.appendChild(createElement("table", null, tbody));
            }

            const homophily = entry.homophily || [];
            if (homophily.length) {
                // One query was recorded per (node_type, edge_type) so the
                // disclosure aggregates across all edge types in this card.
                const homHeader = createElement("div", { className: "block-header" });
                homHeader.appendChild(createElement("h4", { text: "Homophily" }));
                const homDisc = renderQueryDisclosureByPrefix(
                    queriesMap, "nc_supervision:homophily:" + nt + ":"
                );
                if (homDisc) homHeader.appendChild(homDisc);
                card.appendChild(homHeader);
                const tbody = createElement("tbody");
                tbody.appendChild(createElement("tr", null,
                    createElement("th", { text: "Edge type" }),
                    createElement("th", { text: "Edge homophily" }),
                    createElement("th", { text: "Adjusted homophily" }),
                    createElement("th", { text: "Sample size" })));
                for (const h of homophily) {
                    tbody.appendChild(createElement("tr", null,
                        createElement("td", { text: h.edge_type }),
                        createElement("td", {
                            className: "numeric",
                            text: (h.edge_homophily || 0).toFixed(4),
                        }),
                        createElement("td", {
                            className: "numeric",
                            text: (h.adjusted_homophily || 0).toFixed(4),
                        }),
                        createElement("td", {
                            className: "numeric",
                            text: formatNumber(h.edge_sample_count || 0),
                        })));
                }
                card.appendChild(createElement("table", null, tbody));
            }

            const split = entry.cross_split_overlap;
            if (split) {
                card.appendChild(renderBlockHeader(
                    "h4", "Train / val / test split", queriesMap,
                    "nc_supervision:cross_split:" + nt
                ));
                const overlap = split.overlap_node_count || 0;
                const overlapClass = overlap === 0 ? "status-green" : "status-red";
                const overlapCell = createElement("td", { className: "numeric data-value" });
                overlapCell.appendChild(createElement("span", {
                    className: overlapClass,
                    text: formatNumber(overlap),
                }));
                const tbody = createElement("tbody");
                tbody.appendChild(createElement("tr", null,
                    createElement("td", { text: "Cross-split node-id overlap (must be 0)" }),
                    overlapCell));
                for (const [splitValue, count] of Object.entries(split.split_value_counts || {})) {
                    tbody.appendChild(createElement("tr", null,
                        createElement("td", { text: "Rows in split " + JSON.stringify(splitValue) }),
                        createElement("td", {
                            className: "numeric data-value",
                            text: formatNumber(count || 0),
                        })));
                }
                card.appendChild(createElement("table", null, tbody));
            }

            container.appendChild(card);
        }
    }

    function renderAdvanced(analysis, queriesMap) {
        const section = document.getElementById("advanced");
        const container = document.getElementById("advanced-container");

        const classImb = analysis.class_imbalance || {};
        const labelCov = analysis.label_coverage || {};
        const edgeDist = analysis.edge_type_distribution || {};
        const reciprocity = analysis.reciprocity || {};
        const powerLaw = analysis.power_law_exponent || {};

        const hasTier3 = Object.keys(classImb).length ||
                         Object.keys(labelCov).length ||
                         Object.keys(edgeDist).length;
        const hasTier4 = Object.keys(reciprocity).length ||
                         Object.keys(powerLaw).length;

        if (!hasTier3 && !hasTier4) {
            section.hidden = true;
            return;
        }
        section.hidden = false;

        if (Object.keys(classImb).length) {
            const classImbHeader = createElement("div", { className: "block-header" });
            classImbHeader.appendChild(createElement("h3", { text: "Class imbalance" }));
            const classImbDisc = renderQueryDisclosureByPrefix(
                queriesMap, "advanced:class_imbalance:"
            );
            if (classImbDisc) classImbHeader.appendChild(classImbDisc);
            container.appendChild(classImbHeader);
            for (const nodeType of Object.keys(classImb)) {
                const counts = classImb[nodeType];
                const values = Object.values(counts);
                const maxC = Math.max(...values);
                const minC = Math.max(1, Math.min(...values));
                const ratio = maxC / minC;
                const cls = ratio > 10 ? "status-red" : ratio > 5 ? "status-yellow" : "status-green";
                container.appendChild(createElement("p", { className: "data-value" },
                    nodeType + " max/min ratio = ",
                    createElement("span", { className: cls, text: "1:" + ratio.toFixed(1) })));
                const tbody = createElement("tbody");
                for (const [label, count] of Object.entries(counts)) {
                    tbody.appendChild(createElement("tr", null,
                        createElement("td", { text: String(label) }),
                        createElement("td", { className: "numeric data-value", text: formatNumber(count) })));
                }
                container.appendChild(createElement("table", null,
                    createElement("thead", null, createElement("tr", null,
                        createElement("th", { text: "Class" }),
                        createElement("th", { text: "Count" }))),
                    tbody));
            }
        }

        if (Object.keys(labelCov).length) {
            const labelCovHeader = createElement("div", { className: "block-header" });
            labelCovHeader.appendChild(createElement("h3", { text: "Label coverage" }));
            const labelCovDisc = renderQueryDisclosureByPrefix(
                queriesMap, "advanced:label_coverage:"
            );
            if (labelCovDisc) labelCovHeader.appendChild(labelCovDisc);
            container.appendChild(labelCovHeader);
            const tbody = createElement("tbody");
            for (const [nodeType, frac] of Object.entries(labelCov)) {
                tbody.appendChild(createElement("tr", null,
                    createElement("td", { text: nodeType }),
                    createElement("td", { className: "numeric data-value", text: formatPercent(frac) })));
            }
            container.appendChild(createElement("table", null,
                createElement("thead", null, createElement("tr", null,
                    createElement("th", { text: "Node type" }),
                    createElement("th", { text: "Coverage" }))),
                tbody));
        }

        if (Object.keys(edgeDist).length) {
            const edgeDistHeader = createElement("div", { className: "block-header" });
            edgeDistHeader.appendChild(createElement(
                "h3", { text: "Edge type distribution" }
            ));
            const edgeDistDisc = renderQueryDisclosureByPrefix(
                queriesMap, "advanced:edge_type_distribution:"
            );
            if (edgeDistDisc) edgeDistHeader.appendChild(edgeDistDisc);
            container.appendChild(edgeDistHeader);
            const total = sumValues(edgeDist);
            const tbody = createElement("tbody");
            for (const [edgeType, count] of Object.entries(edgeDist)) {
                const frac = total > 0 ? count / total : 0;
                let cls = "status-green";
                if (frac < 0.001) cls = "status-red";
                else if (frac > 0.9) cls = "status-red";
                else if (frac > 0.8) cls = "status-yellow";
                tbody.appendChild(createElement("tr", null,
                    createElement("td", { text: edgeType }),
                    createElement("td", { className: "numeric data-value", text: formatNumber(count) }),
                    createElement("td", { className: "numeric" },
                        createElement("span", { className: cls, text: formatPercent(frac) }))));
            }
            container.appendChild(createElement("table", null,
                createElement("thead", null, createElement("tr", null,
                    createElement("th", { text: "Edge type" }),
                    createElement("th", { text: "Count" }),
                    createElement("th", { text: "Share" }))),
                tbody));
        }

        if (Object.keys(reciprocity).length) {
            container.appendChild(createElement("h3", { text: "Reciprocity" }));
            const tbody = createElement("tbody");
            for (const [edgeType, val] of Object.entries(reciprocity)) {
                tbody.appendChild(createElement("tr", null,
                    createElement("td", { text: edgeType }),
                    createElement("td", { className: "numeric data-value", text: formatPercent(val) })));
            }
            container.appendChild(createElement("table", null,
                createElement("thead", null, createElement("tr", null,
                    createElement("th", { text: "Edge type" }),
                    createElement("th", { text: "Reciprocity" }))),
                tbody));
        }

        if (Object.keys(powerLaw).length) {
            container.appendChild(createElement("h3", { text: "Power-law exponent" }));
            const tbody = createElement("tbody");
            for (const [edgeType, alpha] of Object.entries(powerLaw)) {
                const cls = alpha < 2 ? "status-red" : alpha < 2.5 ? "status-yellow" : "status-green";
                tbody.appendChild(createElement("tr", null,
                    createElement("td", { text: edgeType }),
                    createElement("td", { className: "numeric" },
                        createElement("span", { className: cls, text: alpha.toFixed(2) }))));
            }
            container.appendChild(createElement("table", null,
                createElement("thead", null, createElement("tr", null,
                    createElement("th", { text: "Edge type" }),
                    createElement("th", { text: "Alpha" }))),
                tbody));
        }
    }

    function renderFooter(analysis, profile) {
        const container = document.getElementById("footer-container");

        // facets_html_paths / stats_paths are list-valued so wide tables can
        // contribute one entry per chunk. Flatten with a "(chunk i/N)" suffix
        // when a table has more than one entry; preserve the legacy unsuffixed
        // form for single-chunk tables (the common case).
        function pushFlattened(label, dict) {
            for (const [k, v] of Object.entries(dict)) {
                const list = Array.isArray(v) ? v : (v ? [v] : []);
                list.forEach((p, i) => {
                    const suffix = list.length > 1
                        ? " (chunk " + (i + 1) + "/" + list.length + ")"
                        : "";
                    artifacts.push(label + " " + k + suffix + ": " + p);
                });
            }
        }

        const artifacts = [];
        pushFlattened("FACETS", (profile && profile.facets_html_paths) || {});
        pushFlattened("Stats", (profile && profile.stats_paths) || {});

        if (artifacts.length) {
            container.appendChild(createElement("h3", { text: "Raw artifacts" }));
            const ul = createElement("ul", { className: "footer-list" });
            for (const a of artifacts) {
                ul.appendChild(createElement("li", null, createElement("code", { text: a })));
            }
            container.appendChild(ul);
        }
    }

    function main() {
        const analysis = parseJSONTag("analysis-data");
        const profile = parseJSONTag("profile-data");
        const queriesMap = (analysis && analysis.queries) || {};
        renderHeader(analysis);
        renderOverview(analysis);
        renderNullRates(analysis, queriesMap);
        renderIntegrity(analysis, queriesMap);
        renderFeatureStatistics(profile);
        renderEmbeddingDiagnostics(profile);
        renderCounts(analysis, queriesMap);
        renderDegree(analysis, queriesMap);
        renderHubs(analysis, queriesMap);
        renderSuperHubWarning(analysis);
        renderNodeClassificationSupervision(analysis, queriesMap);
        renderSupervisionOverlap(analysis, queriesMap);
        renderAdvanced(analysis, queriesMap);
        renderFooter(analysis, profile);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", main);
    } else {
        main();
    }
})();
