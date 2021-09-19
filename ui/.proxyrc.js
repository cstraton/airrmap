// Fix for net::ERR_BLOCKED_BY_RESPONSE.NotSameOriginAfterDefaultedToSameOriginByCoep
// (e.g. when loading tile images)
// https://stackoverflow.com/questions/68663632/stripe-err-blocked-by-response
module.exports = function (app) {
    app.use((req, res, next) => {
        res.removeHeader("Cross-Origin-Resource-Policy")
        res.removeHeader("Cross-Origin-Embedder-Policy")
        next()
    })
}