# m7_bridge.ps1 -- pipe the board's serial port to a TCP socket, nothing more.
#
# The NUCLEO-H753ZI's ST-LINK presents its virtual COM port to WINDOWS. Under WSL2
# the alternative is to pass the whole USB device through with usbipd, which works
# until its service crashes and then needs an administrator to restart it. This
# bridge needs no privilege: it opens the COM port on the Windows side and connects
# to a socket that scripts/m7_host.py is listening on inside WSL, then copies bytes
# both ways until either end closes. Every bit of protocol lives in m7_host.py; this
# file has none, so it never needs to change when the protocol does.
#
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File m7_bridge.ps1 `
#       -Port COM3 -Baud 921600 -TcpPort 47321
#
# m7_host.py starts it; running it by hand is only useful for debugging.

param(
    [string]$Port = 'COM3',
    [int]$Baud = 921600,
    [string]$TcpHost = '127.0.0.1',
    [int]$TcpPort = 47321
)

# Two threads copying two streams. PowerShell script blocks cannot run on threads
# they did not create, so the pump is a few lines of C#, compiled once per run.
Add-Type -TypeDefinition @"
using System;
using System.IO;
using System.IO.Ports;
using System.Net.Sockets;
using System.Threading;

public static class BincvM7Bridge {
    static void Pump(Stream from, Stream to) {
        byte[] buf = new byte[65536];
        try {
            int n;
            while ((n = from.Read(buf, 0, buf.Length)) > 0) {
                to.Write(buf, 0, n);
                to.Flush();
            }
        } catch (Exception) {
        }
    }

    public static int Run(string port, int baud, string host, int tcpPort) {
        SerialPort sp = new SerialPort(port, baud, Parity.None, 8, StopBits.One);
        sp.Handshake = Handshake.None;
        sp.ReadBufferSize = 1 << 20;
        sp.WriteBufferSize = 1 << 20;
        sp.ReadTimeout = SerialPort.InfiniteTimeout;
        sp.WriteTimeout = SerialPort.InfiniteTimeout;
        sp.Open();
        sp.DiscardInBuffer();

        TcpClient client = new TcpClient();
        client.NoDelay = true;
        client.Connect(host, tcpPort);
        NetworkStream ns = client.GetStream();

        Thread toBoard = new Thread(() => Pump(ns, sp.BaseStream));
        Thread toHost = new Thread(() => Pump(sp.BaseStream, ns));
        toBoard.IsBackground = true;
        toHost.IsBackground = true;
        toBoard.Start();
        toHost.Start();
        // The host closing its socket ends the session.
        toBoard.Join();
        sp.Close();
        client.Close();
        return 0;
    }
}
"@

exit [BincvM7Bridge]::Run($Port, $Baud, $TcpHost, $TcpPort)
