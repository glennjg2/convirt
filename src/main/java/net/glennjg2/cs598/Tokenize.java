package net.glennjg2.cs598;

import java.io.*;
import java.util.*;
import java.util.regex.*;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ling.CoreLabel.OutputFormat;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import io.vavr.collection.*;
import io.vavr.collection.List;
import io.vavr.control.*;
import org.apache.commons.lang3.*;

public class Tokenize {

    private static String RawReportsRoot = System.getenv("PWD") + "/dataset/mimic-cxr-reports/files";

    private static String ProcessedReportsRoot = System.getenv("PWD") + "/dataset/processed/mimic-cxr-reports";

    private static StanfordCoreNLP Pipeline = null;

    static {
        var props = new Properties();
        props.setProperty("annotators", "tokenize");
        props.setProperty("coref.algorithm", "neural");
        Pipeline = new StanfordCoreNLP(props);
    }

    public static void main(String[] args) throws IOException {
        // testTokenize();
        //testTokenizeFiles();
        createProcessedReport();
    }

    private static void createProcessedReport() throws IOException {
        var processed = new File(ProcessedReportsRoot + "/reports.csv");
        try (var writer = new PrintWriter(new BufferedWriter(new FileWriter(processed)))) {
            writer.println("subject_id,study_id,report");
            listSubjectDirs().zipWithIndex().forEach(subId_idx -> {
            // listPidDirs().take(5).zipWithIndex().forEach(pid_idx -> {
                var subjectId = subId_idx._1;
                var idx = subId_idx._2;
                if (idx % 1000 == 0) {
                    System.out.println();
                    System.out.println(StringUtils.leftPad(idx.toString(), 5, ' '));
                }
                else if (idx % 100 == 0) {
                    System.out.print(".");
                }
                var reports = listReports(subjectId);
                reports.forEach(r -> {
                    var text = getFindingsImpressions(r).trim();
                    var tokens = tokenize(text);
                    if (tokens.size() > 3 && StringUtils.isNotBlank(text)) {
                        text = StringUtils.replaceChars(text, '"', '\'');
                        if (text.contains(",")) {
                            text = "\"" + text + "\"";
                        }
                        writer.println(subjectId(subjectId) + "," + studyId(r) + "," + text);
                    }
                    /*if (tokens.size() > 3) {
                        text = tokens.map(label -> label.toString(OutputFormat.VALUE)).mkString(" ");
                        if (StringUtils.isNotBlank(text)) {
                            text = StringUtils.replaceChars(text, '"', '\'');
                        }
                        if (text.contains(",")) {
                            text = "\"" + text + "\"";
                        }
                        writer.println(subjectId(subjectId) + "," + studyId(r) + "," + text);
                    }*/
                    /*text = tokens.map(label -> label.toString(OutputFormat.VALUE)).mkString(" ");
                    text = StringUtils.replaceChars(text, '"', '\'');
                    if (text.contains(",")) {
                        text = "\"" + text + "\"";
                    }
                    writer.println(subjectId(subjectId) + "," + studyId(r) + "," + text);*/
                });
            });
        }
    }

    private static String subjectId(File patientDir) {
        return patientDir.getName().substring(1);
    }

    private static String studyId(File report) {
        var id = StringUtils.replace(report.getName(), ".txt", "");
        return id.substring(1);
    }

    private static <T> Option<T> opt(T value) {
        return Option.of(value);
    }

    private static  <T> List<T> list(Iterable<T> iterable) {
        return opt(iterable).map(List::ofAll).getOrElse(List.empty());
    }

    private static  <T> List<T> list(T[] xs) {
        return opt(xs).map(x -> Arrays.asList(x)).map(List::ofAll).getOrElse(List.empty());
    }

    private static List<File> listSubjectDirs() {
        return List.range(10, 20).
            map(i -> new File(RawReportsRoot + "/p" + i)).
            flatMap(f -> list(f.listFiles()).filter(g -> g.isDirectory())).
            sortBy(f -> f.getName());
    }

    private static List<File> listReports(File pidDir) {
        return list(pidDir.listFiles(f -> f.getName().startsWith("s") && f.getName().endsWith(".txt")));
    }

    private static List<CoreLabel> tokenize(String text) {
        var document = new CoreDocument(text);
        Pipeline.annotate(document);
        return List.ofAll(document.tokens());
    }

    private static Pattern AllCapsPattern = Pattern.compile(" *([A-Z][A-Z][A-Z ]*)\\:.*");

    private static String getFindingsImpressions(File report) {
        try {
            var headings = List.of("EXAMINATION", "INDICATION", "TECHNIQUE", "COMPARISON", "RECOMMENDATION", "PORTABLE", "CLINICAL");
            try (var reader = new BufferedReader(new FileReader(report))) {
                var startFindings = false;
                var startImpressions = false;
                var findings = new StringBuffer();
                var impressions = new StringBuffer();
                String line = null;
                while ((line = reader.readLine()) != null) {
                    var trimmedLine = line.trim();
                    var r = line;
                    var matcher = AllCapsPattern.matcher(trimmedLine);
                    /* if (matcher.matches()) {
                        var heading = matcher.group(1);
                        if (!headings.contains(heading) && !heading.contains("FINAL REPORT") && !trimmedLine.startsWith("IMPRESSION:") && !trimmedLine.startsWith("FINDINGS:")) {
                            System.out.println(report.getName() + " - Found heading: " + heading);
                        }
                    } */
                    if (trimmedLine.startsWith("IMPRESSION:")) {
                        startImpressions = true;
                        startFindings = false;
                        if (trimmedLine.length() > "IMPRESSION:".length()) {
                            impressions.append(trimmedLine.replace("IMPRESSION:", "").trim());
                        }
                    }
                    else if (trimmedLine.startsWith("FINDINGS:")) {
                        startFindings = true;
                        startImpressions = false;
                        if (trimmedLine.length() > "FINDINGS:".length()) {
                            findings.append(trimmedLine.replace("FINDINGS:", "").trim());
                        }
                    }
                    else if (startImpressions && StringUtils.isNotBlank(line.trim())) {
                        impressions.append(line);
                    }
                    else if (startFindings && StringUtils.isNotBlank(line.trim())) {
                        findings.append(line);
                    }
                    /* else if (headings.exists(h -> r.trim().startsWith(h)) || r.contains("FINAL REPORT")) {
                        startFindings = false;
                        startImpressions = false;
                    } */
                    else if ((matcher.matches() || r.contains("FINAL REPORT")) && headings.exists(h -> r.trim().startsWith(h))) {
                        startFindings = false;
                        startImpressions = false;
                    }
                }
                return findings.toString() + " " + impressions.toString();
            }
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void testTokenizeFiles() {
        List.of("s50414267.txt", "s53189527.txt", "s53911762.txt", "s56699142.txt").forEach(n -> {
            var f = new File(RawReportsRoot + "/p10/p10000032/" + n);
            System.out.println("Extracted: " + getFindingsImpressions(f));
        });
    }

    private static void testOutput() {
        listSubjectDirs().take(2).map(p -> {
            var rs = listReports(p);
            var fs = rs.map(r -> {
                var f = getFindingsImpressions(r);
                System.out.println(p.getName() + "/" + r.getName() + ": " + f);
                return f;
            });
            return fs;
        });
    }

    private static void testTokenize() {
        var text = "Joe Smith was born in California. " +
            "In 2017, he went to Paris, France in the summer. " +
            "His flight left at 3:00pm on July 10th, 2017. " +
            "After eating some escargot for the first time, Joe said, \"That was delicious!\" " +
            "He sent a postcard to his sister Jane Smith. " +
            "After hearing about Joe's trip, Jane decided she might go to France one day.";

        var props = new Properties();
        props.setProperty("annotators", "tokenize");
        props.setProperty("coref.algorithm", "neural");
        var pipeline = new StanfordCoreNLP(props);
        var document = new CoreDocument(text);
        pipeline.annotate(document);
        System.out.println("Tokens: " + document.tokens());
        System.out.println("10th: " + document.tokens().get(10));
        for (var label : document.tokens()) {
            System.out.println(label.word());
        }

        System.getenv().forEach((k, v) -> System.out.println("env key: " + k + ": " + v));
        System.out.println("-----");
        System.getProperties().forEach((k, v) -> System.out.println("prop key: " + k + ": " + v));
    }
}
