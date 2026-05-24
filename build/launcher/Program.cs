// SC_Toolbox launcher — replaces SC_Toolbox.vbs as the Velopack-compatible
// entry point. Spawns the bundled Python interpreter on skill_launcher.py
// with the same window arguments the .vbs used (100 100 500 550 0.95 nul).
//
// On a Velopack install the layout looks like:
//
//     <install root>\
//         SC_Toolbox.exe              (this file)
//         Update.exe                  (Velopack's update agent)
//         current\
//             python\pythonw.exe      (bundled interpreter)
//             skill_launcher.py       (Python entry point)
//             core\, shared\, ui\, skills\, tools\, ...
//
// AppContext.BaseDirectory points at the install root. The Python tree
// lives under current\ — that's where vpk pack drops everything from
// the staging dir we hand it.
//
// On a dev / direct-staging run the layout is flat:
//
//     <staging root>\
//         SC_Toolbox.exe
//         python\pythonw.exe
//         skill_launcher.py
//         ...
//
// We try the Velopack layout first, fall back to the flat layout.

using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using Velopack;
using Velopack.Sources;

namespace SC_Toolbox;

internal static class Program
{
    // GitHub repo that hosts the .nupkg + RELEASES manifest used by the
    // auto-update flow. Updates ship as deltas (~MBs) instead of a full
    // 1.4 GB redownload, so users on v2.2.6 will receive v2.2.7+ in the
    // background and just need to relaunch.
    private const string UPDATE_REPO_URL =
        "https://github.com/ScPlaceholder/SC-Toolbox-Beta-V2";

    private static int Main(string[] args)
    {
        // VelopackApp.Build().Run() must be the FIRST thing the entry exe
        // does. It handles install / uninstall / update hooks (e.g.
        // --squirrel-firstrun, --squirrel-uninstall) and exits early if
        // the launch is one of those lifecycle events. For a normal user
        // launch it's a no-op and control flows straight through.
        VelopackApp.Build().Run();

        // Fire-and-forget update check on every launch. Runs in the
        // background while the Python app is starting; if a newer release
        // exists on GitHub, it downloads the delta and applies it on the
        // next launch (so the current session keeps running normally).
        // All errors swallowed — auto-update is best-effort, never blocks
        // the user from launching the app.
        _ = Task.Run(async () =>
        {
            try
            {
                var mgr = new UpdateManager(new GithubSource(UPDATE_REPO_URL, null, false));
                if (!mgr.IsInstalled) return;  // dev / portable launch — skip
                var info = await mgr.CheckForUpdatesAsync();
                if (info == null) return;  // already on latest
                await mgr.DownloadUpdatesAsync(info);
                // ApplyUpdatesAndRestart triggers Velopack's swap-on-next-launch.
                // We don't restart immediately — the user is in the middle of
                // using the app. Velopack's ApplyUpdates writes a marker; the
                // swap happens automatically when the user closes and reopens
                // SC_Toolbox.exe.
                mgr.WaitExitThenApplyUpdates(info);
            }
            catch
            {
                // Best-effort — never block launch on update errors.
            }
        });

        string root = AppContext.BaseDirectory;

        // Velopack layout: app files live in current\
        string currentDir = Path.Combine(root, "current");
        string appDir = Directory.Exists(currentDir) ? currentDir : root;

        string pythonw = Path.Combine(appDir, "python", "pythonw.exe");
        string pythonExe = Path.Combine(appDir, "python", "python.exe");
        string script = Path.Combine(appDir, "skill_launcher.py");

        string interpreter = File.Exists(pythonw)
            ? pythonw
            : File.Exists(pythonExe) ? pythonExe : "";

        if (interpreter.Length == 0 || !File.Exists(script))
        {
            // Show a Win32 message box without pulling in WinForms.
            ShowError(
                "Bundled Python or skill_launcher.py not found.\n\n"
              + $"Searched: {appDir}\n\n"
              + "Please reinstall SC Toolbox.");
            return 1;
        }

        var psi = new ProcessStartInfo
        {
            FileName = interpreter,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = appDir,
        };

        psi.ArgumentList.Add(script);

        // Default window placement args, mirroring SC_Toolbox.vbs.
        // If the user passed any args (e.g. Velopack's --squirrel-firstrun),
        // forward them — the launcher script ignores unknown args.
        if (args.Length > 0)
        {
            foreach (var a in args)
                psi.ArgumentList.Add(a);
        }
        else
        {
            psi.ArgumentList.Add("100");
            psi.ArgumentList.Add("100");
            psi.ArgumentList.Add("500");
            psi.ArgumentList.Add("550");
            psi.ArgumentList.Add("0.95");
            psi.ArgumentList.Add("nul");
        }

        try
        {
            using var proc = Process.Start(psi);
            // Don't wait — we want the launcher to exit immediately so
            // Velopack can update us next time without thinking we're running.
            return 0;
        }
        catch (Exception ex)
        {
            ShowError($"Failed to start Python: {ex.Message}");
            return 1;
        }
    }

    [System.Runtime.InteropServices.DllImport("user32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode)]
    private static extern int MessageBoxW(IntPtr hWnd, string text, string caption, uint type);

    private static void ShowError(string message)
    {
        // MB_OK | MB_ICONERROR
        MessageBoxW(IntPtr.Zero, message, "SC Toolbox", 0x10);
    }
}
