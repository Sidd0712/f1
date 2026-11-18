self.addEventListener("push", (e) => {
  const d = e.data ? e.data.json() : { title: "Race Intelligence", body: "", url: "./" };
  e.waitUntil(self.registration.showNotification(d.title, { body: d.body, data: { url: d.url } }));
});
self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  e.waitUntil(clients.openWindow(e.notification.data.url));
});
